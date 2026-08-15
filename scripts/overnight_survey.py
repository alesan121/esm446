#!/usr/bin/env python3
"""Unattended overnight capture: gain sweep (C2) then a long metrics-only run (C3).

Run with the machine's own poetry environment::

    poetry run python scripts/overnight_survey.py

Not part of the esm446 package on purpose -- it is an operator tool for one specific
overnight session on one specific machine, not a library import anyone else's code should
depend on.

Uses `hackrf_transfer` to capture, not `esm446.core.source.SoapySource`. That was tried first
and failed immediately: this machine's SoapySDR Python binding is built for the system's
Python 3.10 (`_SoapySDR.cpython-310-x86_64-linux-gnu.so`), and this project's poetry
environment is Python 3.13 -- an ABI mismatch, not a missing package, and no amount of
symlinking fixes it. `hackrf_transfer` is what every real capture in this project's history
has actually used, so this reuses that rather than a live-streaming path that has apparently
never been exercised against real hardware in this repository at all. Captured files are read
back with `esm446.core.source.FileSource`, which needs no SoapySDR binding.

Guardrails, because this runs with nobody watching it:

- Disk headroom is checked before every phase and every chunk. Below the floor, the current
  phase stops cleanly rather than filling the disk.
- Every capture point gets a sidecar recording the physical setup: antenna, gains, firmware,
  host, start time. A number without its mounting conditions is not a measurement -- the
  11.0 dB detection-probability figure this project retracted this session was exactly that
  mistake, on a file whose own antenna condition was recorded but never checked.
- C2 keeps raw IQ (bounded, ~30 s per point, deleted-nothing). C3 does not: each chunk is
  captured, run through the real detection pipeline immediately, and then deleted, keeping
  only emission events and periodic metrics -- 9+ hours of raw cs8 at 2 MS/s is >100 GB and
  this disk should not be spent on data nobody can load into 8 GB of RAM afterwards.
- A safety shutdown is scheduled before anything else runs, so a hang still ends the session
  rather than running the machine untended indefinitely with nobody there to notice.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esm446.core import bands  # noqa: E402
from esm446.core.channelizer import ChannelizerConfig  # noqa: E402
from esm446.core.detector import CfarConfig  # noqa: E402
from esm446.core.node import EsmNode  # noqa: E402
from esm446.core.rfchain import quantise_gains  # noqa: E402
from esm446.core.source import FileSource  # noqa: E402
from esm446.io.sinks import JsonlSink  # noqa: E402

OUTPUT_ROOT = Path.home() / "esm446_overnight"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_ROOT / "survey.log")],
)
logger = logging.getLogger("overnight_survey")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

MIN_FREE_GB = 50.0
ANTENNA_DESCRIPTION = "telescopic, extended to full length, near window"

SAMPLE_RATE_HZ = 2_000_000.0
CENTRE_HZ = float(bands.DEFAULT_CENTRE_HZ)
NUM_CHANNELS = 160

C2_LNA_VALUES = (0.0, 8.0, 16.0, 24.0, 32.0, 40.0)
C2_VGA_VALUES = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
C2_SECONDS_PER_POINT = 30.0

# Budget for the whole unattended session. Kept short of the true available time so the
# safety shutdown below is a genuine fallback, not the expected path.
TOTAL_BUDGET_S = 9.25 * 3600.0
# C3 runs at the shipped default -- the operating point the rest of this project's figures
# already describe.
C3_LNA_DB = 32.0
C3_VGA_DB = 20.0
C3_CHUNK_SECONDS = 240.0

# Absolute fallback: fires even if this script hangs. Comfortably past TOTAL_BUDGET_S plus
# setup time.
SAFETY_SHUTDOWN_MINUTES = 630


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def check_disk(context: str) -> bool:
    free = free_gb(OUTPUT_ROOT)
    if free < MIN_FREE_GB:
        logger.error(
            "disk guard: %.1f GB free, below the %.0f GB floor -- stopping %s",
            free,
            MIN_FREE_GB,
            context,
        )
        return False
    return True


def firmware_version() -> str:
    try:
        out = subprocess.run(["hackrf_info"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "Firmware Version" in line:
                return line.split(":", 1)[1].strip()
    except Exception as exc:  # noqa: BLE001 -- diagnostic metadata, not load-bearing
        return f"unknown ({exc})"
    return "unknown"


@dataclass
class Mount:
    """The physical setup a number was measured under. Written beside every result."""

    antenna: str
    lna_db: float
    vga_db: float
    amp_enabled: bool
    sample_rate_hz: float
    centre_hz: float
    firmware: str
    started_at: str
    host: str


def mount_for(lna_db: float, vga_db: float) -> Mount:
    import socket

    return Mount(
        antenna=ANTENNA_DESCRIPTION,
        lna_db=lna_db,
        vga_db=vga_db,
        amp_enabled=False,
        sample_rate_hz=SAMPLE_RATE_HZ,
        centre_hz=CENTRE_HZ,
        firmware=firmware_version(),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=socket.gethostname(),
    )


def capture(path: Path, seconds: float, lna_db: float, vga_db: float) -> dict:
    """One hackrf_transfer capture. Returns overrun count and whether it completed cleanly."""
    quantised = quantise_gains(lna_db, vga_db)
    num_samples = int(seconds * SAMPLE_RATE_HZ)
    cmd = [
        "hackrf_transfer",
        "-r",
        str(path),
        "-f",
        str(int(CENTRE_HZ)),
        "-s",
        str(int(SAMPLE_RATE_HZ)),
        "-l",
        str(int(quantised.lna_db)),
        "-g",
        str(int(quantised.vga_db)),
        "-n",
        str(num_samples),
        "-B",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 60)
    overruns = 0
    for line in result.stderr.splitlines():
        if "overruns" in line and "longest" in line:
            try:
                overruns = int(line.strip().split()[0])
            except (ValueError, IndexError):
                pass
    completed = path.exists() and path.stat().st_size > 0
    return {
        "completed": completed,
        "overruns": overruns,
        "returncode": result.returncode,
        "bytes_written": path.stat().st_size if path.exists() else 0,
    }


# --------------------------------------------------------------------------------------
# C2 -- gain sweep. One short raw capture per (LNA, VGA) point, analysed and released
# before the next point opens.
# --------------------------------------------------------------------------------------


def run_c2(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for lna in C2_LNA_VALUES:
        for vga in C2_VGA_VALUES:
            if not check_disk("C2 gain sweep"):
                return {"points": results, "aborted": "disk"}

            quantised = quantise_gains(lna, vga)
            label = f"lna{int(quantised.lna_db):02d}_vga{int(quantised.vga_db):02d}"
            iq_path = out_dir / f"{label}.cs8"
            logger.info("C2: %s", label)

            outcome = capture(iq_path, C2_SECONDS_PER_POINT, lna, vga)
            if not outcome["completed"]:
                logger.error("C2: %s failed to capture (rc=%s)", label, outcome["returncode"])
                continue

            raw = np.fromfile(iq_path, dtype=np.int8)
            code_min, code_max = int(raw.min()), int(raw.max())
            occupied_codes = code_max - code_min + 1
            iq = (raw.astype(np.float32) / 128.0).view(np.complex64)
            noise_floor_linear = float(np.mean(np.abs(iq) ** 2))

            mount = mount_for(lna, vga)
            record = {
                **asdict(mount),
                "label": label,
                "occupied_codes": occupied_codes,
                "approx_bits": float(np.log2(max(occupied_codes, 1))),
                "noise_floor_linear": noise_floor_linear,
                "noise_floor_db": 10 * np.log10(max(noise_floor_linear, 1e-30)),
                **outcome,
            }
            (out_dir / f"{label}.json").write_text(json.dumps(record, indent=2, default=str))
            results.append(record)
            logger.info(
                "C2: %s -> %.1f bits occupied, floor %.1f dB, %d overruns",
                label,
                record["approx_bits"],
                record["noise_floor_db"],
                outcome["overruns"],
            )
            del raw, iq

    return {"points": results, "aborted": None}


# --------------------------------------------------------------------------------------
# C3 -- long run at the shipped default: capture a chunk, process it through the real
# pipeline, log metrics, delete the chunk. No continuous raw IQ kept.
# --------------------------------------------------------------------------------------


def run_c3(out_dir: Path, deadline: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mount.json").write_text(
        json.dumps(asdict(mount_for(C3_LNA_DB, C3_VGA_DB)), indent=2)
    )

    config = ChannelizerConfig(
        sample_rate=SAMPLE_RATE_HZ, num_channels=NUM_CHANNELS, decimation=NUM_CHANNELS // 2
    )
    node = EsmNode(
        channelizer_config=config,
        centre_frequency=CENTRE_HZ,
        cfar_config=CfarConfig(pfa=1e-8, method="os"),
    )
    sink = JsonlSink(out_dir / "emissions.jsonl")
    metrics_path = out_dir / "metrics.jsonl"
    chunk_path = out_dir / "_chunk.cs8"

    started = time.time()
    reports_total = 0
    chunk_index = 0
    total_overruns = 0
    aborted = None

    import resource

    with open(metrics_path, "a") as metrics_handle:
        while True:
            now = time.time()
            remaining = deadline - now
            if remaining <= 0:
                logger.info("C3: time budget spent, stopping cleanly")
                break
            if not check_disk("C3 long run"):
                aborted = "disk"
                break

            chunk_seconds = min(C3_CHUNK_SECONDS, remaining)
            chunk_start = time.time()
            outcome = capture(chunk_path, chunk_seconds, C3_LNA_DB, C3_VGA_DB)
            total_overruns += outcome["overruns"]

            batch_count = 0
            if outcome["completed"]:
                with FileSource(chunk_path, SAMPLE_RATE_HZ, CENTRE_HZ, "cs8") as source:
                    while True:
                        block = source.read(1 << 16)
                        if block is None:
                            break
                        if block.size:
                            batch = node.process_block(block)
                            if batch:
                                sink.write(batch)
                                reports_total += len(batch)
                                batch_count += len(batch)
                final = node.flush()
                if final:
                    sink.write(final)
                    reports_total += len(final)
                    batch_count += len(final)
            else:
                logger.error(
                    "C3: chunk %d capture failed (rc=%s)", chunk_index, outcome["returncode"]
                )

            chunk_path.unlink(missing_ok=True)

            usage = resource.getrusage(resource.RUSAGE_SELF)
            metrics_handle.write(
                json.dumps(
                    {
                        "timestamp": now,
                        "chunk_index": chunk_index,
                        "chunk_seconds": chunk_seconds,
                        "chunk_capture_wall_s": time.time() - chunk_start,
                        "elapsed_s": now - started,
                        "frames_processed": node.frames_processed,
                        "emissions_this_chunk": batch_count,
                        "emissions_total": reports_total,
                        "chunk_overruns": outcome["overruns"],
                        "overruns_total": total_overruns,
                        "chunk_completed": outcome["completed"],
                        "maxrss_kb": usage.ru_maxrss,
                        "free_disk_gb": free_gb(OUTPUT_ROOT),
                    }
                )
                + "\n"
            )
            metrics_handle.flush()
            chunk_index += 1

    return {
        "aborted": aborted,
        "elapsed_s": time.time() - started,
        "chunks": chunk_index,
        "emissions_total": reports_total,
        "overruns_total": total_overruns,
        "frames_processed": node.frames_processed,
    }


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def schedule_safety_shutdown() -> None:
    try:
        subprocess.run(["shutdown", "-h", f"+{SAFETY_SHUTDOWN_MINUTES}"], check=True)
        logger.info("safety shutdown scheduled: +%d minutes", SAFETY_SHUTDOWN_MINUTES)
    except Exception as exc:  # noqa: BLE001
        logger.error("could not schedule safety shutdown: %s -- refusing to run unattended", exc)
        raise SystemExit(1) from exc


def clean_shutdown() -> None:
    subprocess.run(["shutdown", "-c"], check=False)
    subprocess.run(["sync"], check=False)
    logger.info("clean shutdown: now")
    subprocess.run(["shutdown", "-h", "now"], check=False)


def main() -> int:
    status_path = OUTPUT_ROOT / "STATUS.txt"
    status_path.write_text(
        f"RUNNING\nstarted {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
    )

    if not check_disk("startup"):
        status_path.write_text("ABORTED: insufficient disk at startup\n")
        return 1

    schedule_safety_shutdown()

    deadline = time.time() + TOTAL_BUDGET_S
    summary: dict = {}

    try:
        logger.info("=== C2: gain sweep ===")
        summary["c2"] = run_c2(OUTPUT_ROOT / "c2_gain_sweep")

        remaining = deadline - time.time()
        if remaining < 600:
            logger.warning("C2 used almost the whole budget; C3 will be short")

        logger.info(
            "=== C3: long run at LNA %.0f / VGA %.0f, %.1f h remaining ===",
            C3_LNA_DB,
            C3_VGA_DB,
            remaining / 3600,
        )
        summary["c3"] = run_c3(OUTPUT_ROOT / "c3_long_run", deadline)

        status_path.write_text("COMPLETED\n" + json.dumps(summary, indent=2, default=str) + "\n")
        logger.info("overnight survey complete")
        return 0

    except Exception:  # noqa: BLE001
        logger.exception("overnight survey failed")
        status_path.write_text(
            "ABORTED: exception, see survey.log\n"
            + json.dumps(summary, indent=2, default=str)
            + "\n"
        )
        return 1

    finally:
        clean_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
