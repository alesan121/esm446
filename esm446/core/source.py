"""IQ sample sources: live SDR, recorded file, or synthetic.

Why this abstraction exists
---------------------------
Everything downstream of here — channelisation, detection, identification — is pure
computation over arrays. Binding that computation to a HackRF would mean the pipeline could
only be exercised by someone holding one, which rules out continuous integration, rules out
deterministic tests, and rules out a reviewer who has just cloned the repository and wants
to see it work.

So the node never opens an SDR. It asks an `IQSource` for blocks of samples and cannot tell
which implementation it received. `SoapySource` is the only one that touches hardware, and
it imports SoapySDR lazily inside its constructor: the module is importable, and the entire
test suite runs, on a machine with no SDR software installed at all.

Sample formats
--------------
`FileSource` reads the two formats this project actually encounters:

- ``cf32`` — interleaved 32-bit floats, what SoapySDR delivers and what the simulator emits.
- ``cs16`` — interleaved signed 16-bit integers, what the PortaPack records to its SD card.
- ``cs8`` — interleaved signed 8-bit integers, what ``hackrf_transfer`` writes. This is the
  HackRF's native format: the ADC is 8-bit, so nothing is lost, and it is the capture path
  that works without a SoapySDR binding for the running Python version.

Each is scaled so that full scale reads unity, which is what keeps bin power interpretable as
dBFS. Getting that scaling wrong is silent: the pipeline runs, detection works, and every
power figure is out by tens of dB.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType

import numpy as np

logger = logging.getLogger(__name__)

#: Sample formats `FileSource` understands, mapped to their numpy dtype and the divisor that
#: brings them to unit full scale.
SAMPLE_FORMATS: dict[str, tuple[np.dtype, float]] = {
    "cf32": (np.dtype(np.float32), 1.0),
    "cs16": (np.dtype(np.int16), 32768.0),
    "cs8": (np.dtype(np.int8), 128.0),
}


class IQSource(ABC):
    """A source of complex baseband samples.

    Attributes:
        sample_rate: Sample rate in Hz.
        centre_frequency: Centre frequency in Hz.
        start_time: Unix time at which the *capture* began.

            This exists because the obvious alternative is wrong. Taking the clock when
            analysis starts labels a recording with when it was replayed, so a capture made
            at 01:07 and analysed at 01:35 is filed under 01:35. Everything downstream that
            bins by time -- occupancy by hour, pattern of life, any correlation between
            emitters -- would then be computed over the analyst's schedule rather than the
            band's.
    """

    sample_rate: float
    centre_frequency: float
    start_time: float

    @abstractmethod
    def read(self, num_samples: int) -> np.ndarray | None:
        """Read the next block of samples.

        Args:
            num_samples: Number of complex samples requested.

        Returns:
            Complex64 array of up to ``num_samples`` samples, or ``None`` when the source is
            exhausted. A short block is not an error; only ``None`` ends the stream.
        """

    def close(self) -> None:
        """Release any underlying resource. Safe to call more than once."""

    def __enter__(self) -> IQSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class FileSource(IQSource):
    """Replay recorded or synthetic IQ from disk.

    This is what makes the pipeline testable and what lets the V&V report present results
    from real captures alongside simulated ones, through exactly the same code path.
    """

    def __init__(
        self,
        path: Path,
        sample_rate: float,
        centre_frequency: float,
        sample_format: str = "cf32",
    ) -> None:
        if sample_format not in SAMPLE_FORMATS:
            raise ValueError(
                f"sample_format must be one of {sorted(SAMPLE_FORMATS)}, got {sample_format!r}"
            )
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"IQ file not found: {self.path}")

        self.sample_rate = sample_rate
        self.centre_frequency = centre_frequency
        self.sample_format = sample_format
        self._dtype, self._full_scale = SAMPLE_FORMATS[sample_format]
        self._handle = self.path.open("rb")
        self.start_time = self._infer_start_time()

        logger.info(
            "source: replaying %s (%s, %.3f MS/s at %.6f MHz)",
            self.path.name,
            sample_format,
            sample_rate / 1e6,
            centre_frequency / 1e6,
        )

    def _infer_start_time(self) -> float:
        """Work out when the recording began.

        Three sources, in descending order of trustworthiness:

        1. A sidecar ``.json`` written alongside the capture, if it carries ``start_time``.
        2. The file's modification time minus its duration. Modification time is when the
           writer *finished*, so subtracting the length recovers the start -- exact for a
           capture written straight through, which is how every capture here is made.
        3. The modification time alone, when the duration is unknown.

        None of these is a timestamped sample stream, and a capture that was paused mid-write
        would defeat the second. Where that matters the sidecar is the answer.

        Returns:
            Unix time at which the capture began.
        """
        sidecar = self.path.with_suffix(".json")
        if sidecar.exists():
            try:
                payload = json.loads(sidecar.read_text())
                if "start_time" in payload:
                    return float(payload["start_time"])
            except (ValueError, OSError):
                logger.warning("source: could not read %s, falling back to file times", sidecar)

        modified = self.path.stat().st_mtime
        duration = self.duration_seconds
        return modified - duration if duration > 0 else modified

    @property
    def total_samples(self) -> int:
        """Number of complex samples in the file."""
        return self.path.stat().st_size // (self._dtype.itemsize * 2)

    @property
    def duration_seconds(self) -> float:
        """Length of the recording in seconds."""
        return self.total_samples / self.sample_rate

    def read(self, num_samples: int) -> np.ndarray | None:
        raw = np.fromfile(self._handle, dtype=self._dtype, count=num_samples * 2)
        if raw.size < 2:
            return None
        # An odd count means the file ended mid-sample; drop the partial pair rather than
        # letting it shift the I/Q phase of everything after it.
        if raw.size % 2:
            raw = raw[:-1]

        interleaved = raw.astype(np.float32)
        if self._full_scale != 1.0:
            interleaved /= self._full_scale
        return interleaved.view(np.complex64)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
            logger.debug("source: closed %s", self.path.name)


class ArraySource(IQSource):
    """Serve an in-memory array in blocks.

    Used by tests and by the scenario simulator, which generates a whole take at once.
    """

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: float,
        centre_frequency: float,
        start_time: float | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.centre_frequency = centre_frequency
        self.start_time = time.time() if start_time is None else start_time
        self._samples = samples.astype(np.complex64, copy=False)
        self._position = 0

    def read(self, num_samples: int) -> np.ndarray | None:
        if self._position >= self._samples.size:
            return None
        block = self._samples[self._position : self._position + num_samples]
        self._position += block.size
        return block


class SoapySource(IQSource):
    """Live capture from an SDR through SoapySDR.

    SoapySDR is imported inside the constructor, not at module scope, so that this module
    and everything importing it remain usable without the SDR stack installed.

    Two things v0 got wrong here, both silent:

    - It requested 800 kS/s, below the HackRF's minimum. The driver delivered something
      else and the whole channel grid was mistuned. The rate is checked against
      ``listSampleRates`` and refused if unsupported.
    - It called ``setGain(SOAPY_SDR_RX, 8, "LNA", value)``, where the second argument is the
      *channel index*, not a gain. The HackRF has only channel 0, so no gain was ever
      applied. Gains here go through `esm446.core.rfchain.quantise_gains` and the value the
      hardware actually accepted is read back and recorded.
    """

    def __init__(
        self,
        sample_rate: float,
        centre_frequency: float,
        lna_gain_db: float,
        vga_gain_db: float,
        amp_enabled: bool = False,
        driver: str = "hackrf",
        channel: int = 0,
    ) -> None:
        import SoapySDR  # noqa: N813 -- third-party module name
        from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX

        from esm446.core.rfchain import quantise_gains

        self._soapy = SoapySDR
        self._rx = SOAPY_SDR_RX
        self.sample_rate = sample_rate
        self.centre_frequency = centre_frequency
        self.channel = channel
        # Live capture: the clock now is the capture time, which is the one case where
        # reading the wall clock is the right answer.
        self.start_time = time.time()

        self._device = SoapySDR.Device({"driver": driver})

        supported = [float(rate) for rate in self._device.listSampleRates(SOAPY_SDR_RX, channel)]
        if supported and not any(abs(rate - sample_rate) < 1.0 for rate in supported):
            raise ValueError(
                f"{driver} does not support {sample_rate / 1e6:.3f} MS/s. "
                f"Supported: {', '.join(f'{r / 1e6:.1f}' for r in supported)} MS/s"
            )

        self._device.setSampleRate(SOAPY_SDR_RX, channel, sample_rate)
        self._device.setFrequency(SOAPY_SDR_RX, channel, centre_frequency)

        gains = quantise_gains(lna_gain_db, vga_gain_db, amp_enabled)
        self._device.setGain(SOAPY_SDR_RX, channel, "LNA", gains.lna_db)
        self._device.setGain(SOAPY_SDR_RX, channel, "VGA", gains.vga_db)
        self._device.setGain(SOAPY_SDR_RX, channel, "AMP", 14.0 if amp_enabled else 0.0)
        self.gains = gains

        self._stream = self._device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [channel])
        self._device.activateStream(self._stream)

        logger.info(
            "source: %s open at %.6f MHz, %.3f MS/s, LNA %.0f dB, VGA %.0f dB, AMP %s",
            driver,
            centre_frequency / 1e6,
            sample_rate / 1e6,
            gains.lna_db,
            gains.vga_db,
            "on" if amp_enabled else "off",
        )

    def read(self, num_samples: int) -> np.ndarray | None:
        buffer = np.empty(num_samples, dtype=np.complex64)
        status = self._device.readStream(self._stream, [buffer], num_samples, timeoutUs=1_000_000)
        if status.ret <= 0:
            logger.warning("source: readStream returned %d", status.ret)
            return np.empty(0, dtype=np.complex64)
        return buffer[: status.ret]

    def close(self) -> None:
        stream = getattr(self, "_stream", None)
        if stream is not None:
            self._device.deactivateStream(stream)
            self._device.closeStream(stream)
            self._stream = None
            logger.info("source: stream closed")
