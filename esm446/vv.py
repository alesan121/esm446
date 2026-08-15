"""Regenerate the verification evidence: every figure in the V&V report, from scratch.

The report is only worth reading if its figures come from the system rather than from a
drawing package. Everything here runs the real channeliser, the real detector and the real
estimator, so a regression changes the picture — and a figure nobody can regenerate is a
claim, not evidence.

One command, no hardware::

    poetry run esm446-vv

Written to ``docs/figures/``. The numbers behind them go to ``docs/figures/results.json`` so
the report's tables and its plots cannot disagree.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from esm446.core.channelizer import ChannelizerConfig, PolyphaseChannelizer
from esm446.core.detector import CfarConfig, CfarDetector
from esm446.core.geolocation import PropagationPrior, estimate_range, path_loss_db

logger = logging.getLogger(__name__)

#: Where the figures land.
FIGURES = Path("docs/figures")

#: Receiver geometry every figure is computed at, matching the shipped configuration.
CONFIG = ChannelizerConfig(sample_rate=2_000_000.0, num_channels=160, decimation=80)

#: Processing gain of the channeliser, 2 MHz into 12.5 kHz. Every SNR in this module is
#: in-channel, because a wideband figure would flatter the system by exactly this much.
PROCESSING_GAIN_DB = 10.0 * np.log10(CONFIG.num_channels)

FREQUENCY_HZ = 446_093_750.0

#: Benchmark repetitions behind the throughput figure. Wall-clock timing of this pipeline
#: varies by tens of per cent with machine load, so the reported number is a median and the
#: observed range is published beside it.
_BENCHMARK_RUNS = 5


def _style(axis: Any, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the one consistent look every figure uses."""
    axis.set_title(title, fontsize=11)
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(True, alpha=0.3, linewidth=0.5)
    axis.tick_params(labelsize=8)


def _tone(offset_bins: float, snr_db: float | None = None, seed: int = 0) -> np.ndarray:
    """A tone offset from bin 20, optionally in noise at a given in-channel SNR."""
    samples = 1 << 17
    t = np.arange(samples) / CONFIG.sample_rate
    frequency = (20 + offset_bins) * CONFIG.channel_spacing
    if snr_db is None:
        return np.exp(2j * np.pi * frequency * t).astype(np.complex64)

    rng = np.random.default_rng(seed)
    amplitude = 3e-3 * 10 ** ((snr_db - PROCESSING_GAIN_DB) / 20)
    noise = 3e-3 * (rng.standard_normal(samples) + 1j * rng.standard_normal(samples)) / np.sqrt(2)
    return (amplitude * np.exp(2j * np.pi * frequency * t) + noise).astype(np.complex64)


def figure_channel_response(results: dict[str, Any]) -> None:
    """Prototype filter response, and what it means for an adjacent channel."""
    import matplotlib.pyplot as plt

    channelizer = PolyphaseChannelizer(CONFIG)
    frequencies, magnitude_db = channelizer.frequency_response()
    offsets_khz = (frequencies - CONFIG.sample_rate / 2) / 1e3

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(frequencies / 1e3, magnitude_db, linewidth=1.2)
    axis.axvline(CONFIG.channel_spacing / 2e3, color="tab:red", linestyle="--", linewidth=1)
    axis.axvline(CONFIG.channel_spacing / 1e3, color="tab:orange", linestyle=":", linewidth=1)
    axis.text(
        CONFIG.channel_spacing / 2e3 + 0.5, -20, "channel edge\n(-6 dB by design)", fontsize=7
    )
    axis.text(CONFIG.channel_spacing / 1e3 + 0.5, -70, "adjacent\nchannel centre", fontsize=7)
    axis.set_xlim(0, 40)
    axis.set_ylim(-120, 5)
    _style(
        axis,
        "REQ-FUN-002 — prototype filter response",
        "offset from channel centre (kHz)",
        "response (dB)",
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "01_channel_response.png", dpi=130)
    plt.close(figure)

    adjacent = float(np.interp(CONFIG.channel_spacing, frequencies, magnitude_db))
    results["adjacent_channel_rejection_db"] = round(-adjacent, 1)
    del offsets_khz


def figure_sensitivity_ripple(results: dict[str, Any]) -> None:
    """Response against offset from a bin centre: the flat top and the cliff."""
    import matplotlib.pyplot as plt

    offsets = np.arange(0.0, 0.501, 0.025)
    peak_db, pair_db = [], []
    reference = None
    for offset in offsets:
        spectra = PolyphaseChannelizer(CONFIG).process(_tone(offset))
        power = (np.abs(spectra[spectra.shape[0] // 4 :]) ** 2).mean(axis=0)
        ordered = np.sort(power)[::-1]
        reference = ordered[0] if reference is None else reference
        peak_db.append(10 * np.log10(ordered[0] / reference))
        pair_db.append(10 * np.log10((ordered[0] + ordered[1]) / reference))

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(offsets, peak_db, "o-", markersize=3, linewidth=1.2, label="peak bin")
    axis.plot(offsets, pair_db, "s--", markersize=3, linewidth=1.2, label="summed pair")
    axis.axhline(-6.02, color="tab:red", linestyle=":", linewidth=1)
    axis.text(0.02, -5.6, "-6.02 dB worst case", fontsize=7, color="tab:red")
    axis.legend(fontsize=8)
    _style(
        axis,
        "REQ-FUN-004 — sensitivity against offset from a bin centre",
        "offset from bin centre (bins)",
        "response relative to on-centre (dB)",
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "02_sensitivity_ripple.png", dpi=130)
    plt.close(figure)

    results["worst_case_scalloping_db"] = round(float(peak_db[-1]), 2)
    results["worst_case_pair_db"] = round(float(pair_db[-1]), 2)


def figure_false_alarm_rate(results: dict[str, Any]) -> None:
    """The claim CFAR makes: the rate holds whatever the noise level."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1234)
    levels = np.logspace(-6, 3, 10)
    design = 1e-3
    measured = {}

    for method in ("ca", "os"):
        rates = []
        detector = CfarDetector(CfarConfig(pfa=design, method=method, update_interval=1))
        for level in levels:
            power = rng.exponential(level, size=(3000, CONFIG.num_channels))
            rates.append(float(detector.detection_mask(power).mean()))
        measured[method] = rates

    figure, axis = plt.subplots(figsize=(7, 4))
    for method, rates in measured.items():
        axis.semilogx(
            levels, rates, "o-", markersize=3, linewidth=1.2, label=f"{method.upper()}-CFAR"
        )
    axis.axhline(design, color="tab:red", linestyle="--", linewidth=1, label="design point")
    axis.set_ylim(0, 2 * design)
    axis.legend(fontsize=8)
    _style(
        axis,
        "REQ-FUN-003 — false alarm rate against noise level, nine orders of magnitude",
        "noise power (arbitrary linear units)",
        "measured probability of false alarm",
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "03_false_alarm_rate.png", dpi=130)
    plt.close(figure)

    results["pfa_design"] = design
    results["pfa_measured_span"] = [
        round(float(min(min(r) for r in measured.values())), 6),
        round(float(max(max(r) for r in measured.values())), 6),
    ]


#: Real receiver noise, antenna disconnected -- see tests/test_false_alarm_on_real_noise.py.
#: Same vector the shipped-default zero-false-alarm result uses, so the two are the same
#: operating point measured two ways: does nothing fire on it, and does a real signal cut
#: through it at the SNR the synthetic-noise curve predicts.
_REAL_NOISE_PATH = Path("tests/data/receiver_noise_lna32_vga20.cs8")


def _real_noise_iq() -> np.ndarray | None:
    """Raw IQ from the committed receiver-noise vector, or None if it is not present."""
    if not _REAL_NOISE_PATH.exists():
        return None
    raw = np.fromfile(_REAL_NOISE_PATH, dtype=np.int8)
    return (raw.astype(np.float32) / 128.0).view(np.complex64)


def _tone_gain(window: int) -> float:
    """Channel-20 power produced by a unit-amplitude tone, for converting SNR to amplitude.

    Calibrated once against the channeliser itself rather than assumed, so it is exactly
    right for this configuration instead of approximately right from a formula.
    """
    t = np.arange(window) / CONFIG.sample_rate
    tone = np.exp(2j * np.pi * 20 * CONFIG.channel_spacing * t).astype(np.complex64)
    spectra = PolyphaseChannelizer(CONFIG).process(tone)
    power = (np.abs(spectra[1000:]) ** 2).astype(np.float64)
    return float(power[:, 20].mean())


def _real_noise_floor(noise: np.ndarray, start: int, window: int) -> float:
    """This window's own in-channel noise floor, away from the DC spur and the target bins.

    Computed from the real receiver noise alone, before any tone is added -- the reference
    the injected tone's SNR is measured against, not a wideband time-domain RMS. Wideband RMS
    would be dominated by the local-oscillator spur at bin 0 (28 dB over the floor, per
    tests/test_false_alarm_on_real_noise.py) and so would systematically overstate the noise
    the target channel actually carries, making every SNR in this sweep too generous.

    Read from bins 6-34, the same reference-cell span `CfarConfig`'s defaults (24 reference
    cells, 2 guard either side) actually use to threshold channel 20 -- not some other, more
    convenient part of the spectrum. A receiver's noise floor is not obliged to be flat with
    frequency, and referencing SNR against a different span than the detector's own threshold
    sees would make the two numbers describe different things while looking like the same
    measurement.
    """
    segment = noise[start : start + window].astype(np.complex64)
    spectra = PolyphaseChannelizer(CONFIG).process(segment)
    power = (np.abs(spectra[1000:]) ** 2).astype(np.float64)
    reference_bins = np.r_[6:18, 22:34]
    return float(np.median(power[:, reference_bins]))


def _real_noise_trial(
    noise: np.ndarray,
    start: int,
    window: int,
    offset_bins: float,
    snr_db: float,
    noise_floor: float,
    gain: float,
) -> np.ndarray:
    """A window of real receiver noise with a synthetic tone added at a controlled in-channel SNR.

    The tone's amplitude is solved from this window's own measured noise floor and the
    channeliser's own measured gain, not a formula carried over from the synthetic case --
    the point is to ask whether the detector reaches the SNR the synthetic curve predicts
    against noise this receiver actually produced, non-Gaussian tails and all.
    """
    segment = noise[start : start + window]
    t = np.arange(window) / CONFIG.sample_rate
    frequency = (20 + offset_bins) * CONFIG.channel_spacing
    target_power = (10 ** (snr_db / 10)) * noise_floor
    amplitude = math.sqrt(target_power / gain)
    tone = amplitude * np.exp(2j * np.pi * frequency * t)
    return (segment + tone).astype(np.complex64)


def figure_detection_probability(results: dict[str, Any]) -> None:
    """Probability of detection against SNR, on synthetic noise and on real receiver noise.

    The synthetic curve alone answers "does the CFAR math work". It cannot answer whether real
    receiver noise costs anything beyond what Pfa already measures, because Pfa alone cannot
    distinguish a well-tuned detector from a deaf one -- both report zero false alarms. This
    repeats the same sweep against real, captured receiver noise (antenna disconnected, the
    same vector the shipped-default zero-false-alarm result uses) with the tone added on top
    at a controlled SNR, so a gap between the two curves is the real cost the synthetic figure
    cannot see.
    """
    import matplotlib.pyplot as plt

    snrs = np.arange(8, 26, 1.0)
    curves: dict[str, list[float]] = {}

    for label, offset in (("on bin centre", 0.0), ("half a bin off centre", 0.5)):
        detected = []
        for index, snr in enumerate(snrs):
            spectra = PolyphaseChannelizer(CONFIG).process(_tone(offset, snr, seed=index))
            power = (np.abs(spectra[1000:]) ** 2).astype(np.float64)
            detector = CfarDetector(CfarConfig(pfa=1e-8, method="os"))
            mask = detector.detection_mask(power)
            detected.append(float(mask[:, 19:23].any(axis=1).mean()))
        curves[label] = detected

    real_noise = _real_noise_iq()
    real_curves: dict[str, list[float]] = {}
    if real_noise is not None:
        window = 1 << 17
        starts = list(range(0, len(real_noise) - window, 40_000))
        real_snrs = np.arange(8, 31, 1.0)
        gain = _tone_gain(window)
        noise_floors = [_real_noise_floor(real_noise, start, window) for start in starts]
        for label, offset in (("on bin centre", 0.0), ("half a bin off centre", 0.5)):
            detected = []
            for snr in real_snrs:
                hits = []
                for start, noise_floor in zip(starts, noise_floors):
                    trial = _real_noise_trial(
                        real_noise, start, window, offset, float(snr), noise_floor, gain
                    )
                    spectra = PolyphaseChannelizer(CONFIG).process(trial)
                    power = (np.abs(spectra[1000:]) ** 2).astype(np.float64)
                    detector = CfarDetector(CfarConfig(pfa=1e-8, method="os"))
                    mask = detector.detection_mask(power)
                    hits.append(bool(mask[:, 19:23].any(axis=1).any()))
                detected.append(float(np.mean(hits)))
            real_curves[label] = detected
        results["real_noise_trials"] = len(starts)

    figure, axis = plt.subplots(figsize=(7, 4))
    colours = {"on bin centre": "tab:blue", "half a bin off centre": "tab:orange"}
    for label, values in curves.items():
        axis.plot(
            snrs,
            values,
            "o-",
            markersize=3,
            linewidth=1.2,
            color=colours[label],
            label=f"{label} (synthetic)",
        )
    for label, values in real_curves.items():
        axis.plot(
            real_snrs,
            values,
            "s--",
            markersize=3,
            linewidth=1.2,
            color=colours[label],
            label=f"{label} (real noise)",
        )
    axis.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    axis.set_ylim(-0.02, 1.02)
    axis.legend(fontsize=7)
    _style(
        axis,
        "REQ-FUN-004 — probability of detection at P_fa 1e-8",
        "in-channel SNR (dB)",
        "probability of detection",
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "04_detection_probability.png", dpi=130)
    plt.close(figure)

    for label, values in real_curves.items():
        crossing = next((real_snrs[i] for i, v in enumerate(values) if v >= 0.5), float("nan"))
        key = "centred" if "centre" in label and "off" not in label else "offset"
        results[f"pd50_real_noise_{key}_db"] = (
            None if np.isnan(crossing) else round(float(crossing), 1)
        )

    for label, values in curves.items():
        crossing = next((snrs[i] for i, v in enumerate(values) if v >= 0.5), float("nan"))
        results[
            f"pd50_{'centred' if 'centre' in label and 'off' not in label else 'offset'}_db"
        ] = round(float(crossing), 1)


def figure_interval_coverage(results: dict[str, Any]) -> None:
    """Do the credible rings contain the truth as often as they say."""
    import matplotlib.pyplot as plt

    prior = PropagationPrior()
    rng = np.random.default_rng(2024)
    percentiles = (5, 50, 68, 90, 95)
    trials = 300
    contained = dict.fromkeys(percentiles, 0)

    for trial in range(trials):
        exponent = max(rng.normal(prior.path_loss_exponent, prior.path_loss_exponent_sigma), 2.0)
        eirp_dbm = rng.normal(prior.eirp_dbm, prior.eirp_sigma_db)
        shadowing_db = rng.normal(0.0, prior.shadowing_sigma_db)
        calibration_db = rng.normal(0.0, prior.calibration_sigma_db)
        truth_m = float(rng.uniform(50.0, 2_000.0))
        received_dbm = (
            eirp_dbm - path_loss_db(truth_m, FREQUENCY_HZ, exponent) - shadowing_db + calibration_db
        )
        estimate = estimate_range(
            received_dbm, FREQUENCY_HZ, prior=prior, draws=4_000, seed=trial, calibrated=True
        )
        for percentile in percentiles:
            contained[percentile] += truth_m <= estimate.ring(percentile)

    achieved = [100.0 * contained[p] / trials for p in percentiles]

    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot([0, 100], [0, 100], color="grey", linestyle="--", linewidth=1, label="ideal")
    axis.plot(percentiles, achieved, "o-", markersize=5, linewidth=1.4, label="measured")
    for p, a in zip(percentiles, achieved, strict=True):
        axis.annotate(f"{a:.0f}%", (p, a), textcoords="offset points", xytext=(6, -10), fontsize=7)
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.legend(fontsize=8)
    _style(
        axis,
        "REQ-CAL-005 — credible interval coverage",
        "declared percentile (%)",
        "fraction of realisations containing the truth (%)",
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "05_interval_coverage.png", dpi=130)
    plt.close(figure)

    results["coverage"] = {str(p): round(a, 1) for p, a in zip(percentiles, achieved, strict=True)}
    results["coverage_trials"] = trials


def figure_misspecification(results: dict[str, Any]) -> None:
    """What the coverage figure cannot tell you: how it fails when the model is wrong.

    The coverage study draws its realisations from the prior the estimator assumes, so it
    verifies the arithmetic and nothing about the environment. This sweeps the *true* path
    loss exponent away from the assumed one and measures what the rings then achieve.
    """
    import matplotlib.pyplot as plt

    prior = PropagationPrior()
    exponents = np.arange(2.5, 4.6, 0.25)
    trials = 200
    achieved: dict[int, list[float]] = {50: [], 95: []}

    for exponent in exponents:
        rng = np.random.default_rng(7)
        contained = dict.fromkeys(achieved, 0)
        for trial in range(trials):
            eirp_dbm = prior.eirp_dbm
            shadowing_db = rng.normal(0.0, prior.shadowing_sigma_db)
            truth_m = float(rng.uniform(50.0, 2_000.0))
            received_dbm = eirp_dbm - path_loss_db(truth_m, FREQUENCY_HZ, exponent) - shadowing_db
            estimate = estimate_range(
                received_dbm, FREQUENCY_HZ, prior=prior, draws=3_000, seed=trial, calibrated=True
            )
            for percentile in contained:
                contained[percentile] += truth_m <= estimate.ring(percentile)
        for percentile, count in contained.items():
            achieved[percentile].append(100.0 * count / trials)

    figure, axis = plt.subplots(figsize=(7, 4.4))
    for percentile, values in achieved.items():
        axis.plot(exponents, values, "o-", markersize=4, linewidth=1.3, label=f"{percentile}% ring")
        axis.axhline(percentile, color="grey", linestyle=":", linewidth=0.8)
    axis.axvline(prior.path_loss_exponent, color="tab:red", linestyle="--", linewidth=1.2)
    axis.text(prior.path_loss_exponent + 0.03, 6, "assumed", fontsize=7, color="tab:red")
    axis.axvspan(exponents[0], prior.path_loss_exponent, color="tab:red", alpha=0.05)
    axis.text(2.6, 88, "environment clearer than assumed:\nrings undercover", fontsize=7)
    axis.set_ylim(-3, 103)
    axis.legend(fontsize=8, loc="lower right")
    _style(
        axis,
        "REQ-CAL-005 — coverage when the environment is not what was assumed",
        "true path loss exponent",
        "realisations containing the truth (%)",
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "07_misspecification.png", dpi=130)
    plt.close(figure)

    results["misspecification"] = {
        f"{e:.2f}": {str(p): round(v[i], 1) for p, v in achieved.items()}
        for i, e in enumerate(exponents)
    }
    worst = min(achieved[95])
    results["worst_95_coverage_under_misspecification"] = round(worst, 1)


def figure_benchmark(results: dict[str, Any]) -> None:
    """Where the time goes, against the v0 baseline.

    Reported as the median of `_BENCHMARK_RUNS` runs. See the comment below for why one run
    is not enough.
    """
    import matplotlib.pyplot as plt

    import statistics

    from esm446.bench import benchmark_node, benchmark_pfb

    # The median of several runs, not one. A single wall-clock measurement of this pipeline
    # varies by up to 45 % with machine load, and quoting one of those as "the" figure is
    # false precision -- it is how the same repository ended up claiming 0.21 in one document
    # and 0.26 in another. The spread is reported alongside so the variance is visible rather
    # than averaged away.
    runs = _BENCHMARK_RUNS
    pfb_ratios = sorted(benchmark_pfb(seconds=2.0).realtime_ratio for _ in range(runs))
    node_ratios = sorted(benchmark_node(seconds=2.0).realtime_ratio for _ in range(runs))
    pfb_ratio = statistics.median(pfb_ratios)
    node_ratio = statistics.median(node_ratios)
    v0_ratio = 6.9

    labels = ["v0 per-channel\nmixer + filter", "polyphase\nfilter bank", "full node\npipeline"]
    ratios = [v0_ratio, pfb_ratio, node_ratio]
    colours = ["tab:red", "tab:blue", "tab:green"]

    figure, axis = plt.subplots(figsize=(6.5, 4))
    bars = axis.bar(labels, ratios, color=colours, width=0.55)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.text(2.35, 1.15, "real time", fontsize=7)
    axis.set_yscale("log")
    for bar, ratio in zip(bars, ratios, strict=True):
        axis.annotate(
            f"{ratio:.2f}",
            (bar.get_x() + bar.get_width() / 2, ratio),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )
    _style(
        axis,
        "REQ-PER-001, REQ-PER-002 — CPU seconds per signal second",
        "",
        "CPU-s per signal second (log scale)",
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "06_throughput.png", dpi=130)
    plt.close(figure)

    results["pfb_cpu_s_per_s"] = round(pfb_ratio, 3)
    results["node_cpu_s_per_s"] = round(node_ratio, 3)
    results["node_cpu_s_per_s_range"] = [round(node_ratios[0], 3), round(node_ratios[-1], 3)]
    results["benchmark_runs"] = runs
    results["v0_cpu_s_per_s"] = v0_ratio


FIGURE_BUILDERS = (
    figure_channel_response,
    figure_sensitivity_ripple,
    figure_false_alarm_rate,
    figure_detection_probability,
    figure_interval_coverage,
    figure_misspecification,
    figure_benchmark,
)


def generate(output_dir: Path = FIGURES) -> dict[str, Any]:
    """Build every figure and return the measurements behind them.

    Args:
        output_dir: Where the PNGs go.

    Returns:
        The measured figures, also written as ``results.json`` beside the plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    global FIGURES  # noqa: PLW0603 -- the builders take their destination from the module
    FIGURES = output_dir

    results: dict[str, Any] = {}
    for builder in FIGURE_BUILDERS:
        logger.info("vv: %s", builder.__name__)
        builder(results)

    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main(argv: list[str] | None = None) -> int:
    """Regenerate the verification figures from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
    logging.getLogger().setLevel(logging.INFO)

    results = generate(args.output)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
