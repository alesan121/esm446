"""The picture of the band, for people who will not read the code.

Two audiences look at a project like this and they need different things in different
amounts of time. An engineer clones it and runs `make demo`. Everybody else looks at one
image and decides whether to keep scrolling. Both are served by the same data, so both are
built here.

`waterfall` renders the band against time with the node's own detections drawn on top, which
is the single frame that shows what the system does: energy arrives, the detector finds it,
the identification names it.

`dashboard` builds a self-contained HTML page from a store of emissions — occupancy by
channel and hour, the emitters the order of battle inferred, and the by-products it declined
to count as emitters. No web server, no external assets, no network: one file that opens in a
browser and can be attached to anything.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from esm446.analysis.artefacts import attribute_products
from esm446.analysis.eob import cluster_emitters, compute_occupancy
from esm446.core import bands
from esm446.core.survey import SpectrumSurvey
from esm446.io.sinks import read_reports

logger = logging.getLogger(__name__)

#: Channels drawn on the waterfall's frequency axis.
_TICK_CHANNELS = (1, 4, 8, 12, 16)


def waterfall(
    iq: np.ndarray,
    sample_rate: float,
    centre_hz: float,
    reports: list[Any],
    path: Path,
    title: str | None = None,
) -> Path:
    """Render the band against time with the node's detections drawn over it.

    Args:
        iq: Complex baseband for the whole capture.
        sample_rate: Sample rate of ``iq``.
        centre_hz: Centre frequency the capture was taken at.
        reports: Emissions the node produced from the same IQ.
        path: Where to write the PNG.
        title: Figure title. Defaults to one naming the capture's real duration, so the
            caption cannot drift from the axis beside it.

    Returns:
        The path written.
    """
    import matplotlib.pyplot as plt

    survey = SpectrumSurvey(sample_rate=sample_rate, centre_hz=centre_hz)
    # Linear power, fftshifted. Decibels for display: a linear waterfall of a band whose
    # emitters differ by 40 dB shows one emitter and a black rectangle.
    with np.errstate(divide="ignore"):
        spectrogram = 10.0 * np.log10(np.maximum(survey.spectrogram(iq), 1e-20))
    duration_s = iq.size / sample_rate

    frequencies_mhz = (
        centre_hz + np.fft.fftshift(np.fft.fftfreq(spectrogram.shape[1], 1.0 / sample_rate))
    ) / 1e6

    # Clip the display range to the noise floor and a little above the strongest emitter.
    # Left alone, the scale runs down to the numerical floor and spends most of its range on
    # decibels nothing occupies.
    floor, ceiling = np.percentile(spectrogram, (5.0, 99.99))

    figure, axis = plt.subplots(figsize=(9, 5))
    image = axis.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=floor - 3.0,
        vmax=ceiling + 3.0,
        extent=(frequencies_mhz[0], frequencies_mhz[-1], 0.0, duration_s),
    )
    figure.colorbar(image, ax=axis, label="power (dBFS)", pad=0.01)

    for report in reports:
        start = report.offset_s
        axis.add_patch(
            plt.Rectangle(
                (report.frequency_hz / 1e6 - 0.00625, start),
                0.0125,
                report.duration_s,
                fill=False,
                edgecolor="white",
                linewidth=1.1,
            )
        )
        label = f"PMR{report.pmr_channel}" if report.pmr_channel else "off-grid"
        if report.ctcss_tone_hz:
            label += f" {report.ctcss_tone_hz:.1f}Hz"
        axis.annotate(
            label,
            (report.frequency_hz / 1e6, start + report.duration_s),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7,
            color="white",
        )

    axis.set_xlim(bands.channel_frequency(1) / 1e6 - 0.05, bands.channel_frequency(16) / 1e6 + 0.05)
    axis.set_xticks([bands.channel_frequency(c) / 1e6 for c in _TICK_CHANNELS])
    axis.set_xticklabels([f"PMR{c}" for c in _TICK_CHANNELS], fontsize=8)
    axis.set_ylabel("time (s)", fontsize=9)
    axis.set_title(
        title or f"PMR446 band \u2014 {duration_s:.0f} s, with the node's detections",
        fontsize=11,
    )
    axis.tick_params(labelsize=8)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)
    logger.info("dashboard: wrote %s", path)
    return path


def _occupancy_grid(occupancy: Any) -> tuple[list[str], list[int], list[list[float]]]:
    """Airtime as a channel-by-hour grid, ready to draw as a table."""
    channels = sorted({c for c, _ in occupancy.airtime_s}, key=lambda c: (c is None, c))
    hours = sorted({h for _, h in occupancy.airtime_s})
    grid = [[occupancy.airtime_s.get((c, h), 0.0) for h in hours] for c in channels]
    labels = [f"PMR{c}" if c is not None else "off-grid" for c in channels]
    return labels, hours, grid


def _cell_colour(seconds: float, peak: float) -> str:
    """Heat for one occupancy cell, as an rgba string."""
    if peak <= 0 or seconds <= 0:
        return "transparent"
    intensity = min(1.0, seconds / peak)
    return f"rgba(220, 80, 40, {0.12 + 0.78 * intensity:.3f})"


def dashboard(reports: list[Any], title: str = "ESM-446 — band picture") -> str:
    """Build a self-contained HTML page from a set of emissions.

    Everything is inlined: no stylesheet, no script, no image request. The page opens from a
    file, survives being emailed, and cannot phone anywhere -- which for a page about
    somebody's radio traffic is the correct property rather than a convenience.

    Args:
        reports: Emissions, typically read back from a store.
        title: Page title.

    Returns:
        The HTML document.
    """
    attributed = attribute_products(reports)
    profiles = cluster_emitters(attributed)
    occupancy = compute_occupancy(attributed)
    labels, hours, grid = _occupancy_grid(occupancy)
    peak = max((max(row) for row in grid), default=0.0)

    products = sum(len(p.products) for p in profiles)
    summary = [
        ("detections", f"{len(attributed)}"),
        ("emitters", f"{len(profiles)}"),
        ("by-products attributed", f"{products}"),
        ("carrier time", f"{occupancy.total_airtime_s:.1f} s"),
        ("observed window", f"{occupancy.window_s / 60:.1f} min"),
        ("band load", f"{occupancy.band_duty_cycle:.1%}"),
    ]

    cards = "".join(
        f'<div class="card"><div class="value">{html.escape(value)}</div>'
        f'<div class="label">{html.escape(label)}</div></div>'
        for label, value in summary
    )

    header = "".join(f"<th>{hour:02d}:00</th>" for hour in hours)
    rows = "".join(
        "<tr><th>{name}</th>{cells}</tr>".format(
            name=html.escape(label),
            cells="".join(
                (
                    f'<td style="background:{_cell_colour(seconds, peak)}">' f"{seconds:.0f}</td>"
                    if seconds
                    else '<td class="empty"></td>'
                )
                for seconds in row
            ),
        )
        for label, row in zip(labels, grid, strict=True)
    )

    emitters = "".join(
        "<tr><td>{label}</td><td>{channel}</td><td>{tone}</td><td>{count}{bound}</td>"
        "<td>{airtime:.1f} s</td><td>{median:.1f} s</td><td>{deviation:.0f} Hz</td>"
        "<td>{products}</td></tr>".format(
            label=html.escape(profile.label),
            channel=f"PMR{profile.pmr_channel}" if profile.pmr_channel else "off-grid",
            tone=f"{profile.ctcss_tone_hz:.1f} Hz" if profile.ctcss_tone_hz else "—",
            count=profile.transmission_count,
            bound=' <span class="bound">≥1</span>' if profile.count_is_lower_bound else "",
            airtime=profile.total_airtime_s,
            median=profile.median_duration_s,
            deviation=profile.median_deviation_hz,
            products=len(profile.products) or "—",
        )
        for profile in profiles
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 2rem 1.5rem 4rem; max-width: 62rem; margin-inline: auto; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 .2rem; }}
  h2 {{ font-size: 1.05rem; margin: 2.2rem 0 .6rem; }}
  .sub {{ opacity: .65; font-size: .9rem; margin-bottom: 1.6rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr));
            gap: .6rem; }}
  .card {{ border: 1px solid rgba(128,128,128,.3); border-radius: .5rem; padding: .7rem .8rem; }}
  .value {{ font-size: 1.45rem; font-weight: 600; }}
  .label {{ font-size: .78rem; opacity: .7; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .84rem; }}
  th, td {{ border: 1px solid rgba(128,128,128,.22); padding: .3rem .5rem; text-align: right; }}
  th {{ font-weight: 600; }}
  tbody th, td:first-child {{ text-align: left; }}
  td.empty {{ opacity: .25; }}
  .bound {{ font-size: .72rem; opacity: .65; }}
  .note {{ font-size: .85rem; opacity: .8; border-left: 3px solid rgba(128,128,128,.35);
           padding-left: .8rem; margin-top: 1rem; }}
  .scroll {{ overflow-x: auto; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="sub">Signal metadata only — never communication content.</div>

<div class="cards">{cards}</div>

<h2>Occupancy — carrier seconds by channel and hour (UTC)</h2>
<div class="scroll"><table><thead><tr><th></th>{header}</tr></thead>
<tbody>{rows}</tbody></table></div>

<h2>Emitters</h2>
<div class="scroll"><table><thead><tr>
<th>emitter</th><th>channel</th><th>CTCSS</th><th>transmissions</th><th>airtime</th>
<th>median</th><th>deviation</th><th>by-products</th>
</tr></thead><tbody>{emitters}</tbody></table></div>

<div class="note">
  Counts marked <span class="bound">≥1</span> are lower bounds: none of that emitter's
  transmissions overlap in time, and a recording cannot separate one radio taking turns from
  several sharing a channel and a sub-audible code.
  By-products are that emitter's own splatter and intermodulation, attributed to it by the
  frequency relation they obey rather than counted as separate emitters.
</div>
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    """Build the dashboard from a store, from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path, help="emission store written by esm446-node")
    parser.add_argument("--output", type=Path, default=Path("out/dashboard.html"))
    parser.add_argument("--title", default="ESM-446 — band picture")
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
    logging.getLogger().setLevel(logging.INFO)

    try:
        reports = read_reports(args.store)
    except (FileNotFoundError, ValueError) as error:
        logger.error("dashboard: %s", error)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dashboard(reports, args.title), encoding="utf-8")
    logger.info("dashboard: wrote %s (%d emissions)", args.output, len(reports))
    print(json.dumps({"output": str(args.output), "emissions": len(reports)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
