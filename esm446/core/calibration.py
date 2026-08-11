"""Conversion from digital power (dBFS) to absolute received power (dBm).

The problem with v0's approach
------------------------------
v0 carried a single constant::

    OFFSET_CAL = -49.3   # "provisional"

and applied it to every measurement. Three things were wrong with that, and only the third
is about the number itself.

**It was not attached to a gain configuration.** Receiver gain sits between the antenna and
the sample, so the offset is only meaningful alongside the LNA, VGA and AMP settings it was
measured at. v0 recorded power without gains, which makes its archive uncalibratable in
retrospect — the information needed to fix it later was never written down. Here a
calibration is keyed by gain configuration and refuses to extrapolate to one it has not
seen.

**It was a single point.** One offset assumes the receiver is perfectly linear across its
whole range. Real front ends compress at the top and disappear into their own noise at the
bottom, and a single point tells you nothing about where those limits are. A swept
calibration gives the slope, the residual, the compression point and the usable dynamic
range — and the slope is itself a check: it must come out at 1.0 dB per dB, and if it does
not, something in the measurement is wrong.

**It was uncalibrated and did not say so.** The docstring called it provisional; the code
emitted distances anyway. Here an uncalibrated system reports ``None`` for dBm and marks
its estimates accordingly, because a number that looks like a measurement and is not is
worse than no number.

Method
------
The measurement is *conducted*: signal generator into coaxial cable into a fixed attenuator
into the receiver. The path loss is the attenuator, which is known and repeatable, instead
of an over-the-air path that has to be estimated — you cannot calibrate a receiver through
an unknown loss. See ``docs/04_link_budget.md`` for the bench setup, and note that
`esm446.core.rfchain.RfChain.check_input_safety` gates the levels involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from esm446.core.rfchain import HackrfGains, quantise_gains

#: Power below which a bin is treated as empty rather than as a very weak signal.
MINIMUM_LINEAR_POWER = 1e-20


def linear_to_dbfs(power_linear: float | np.ndarray) -> float | np.ndarray:
    """Convert linear bin power to dB relative to full scale.

    Full scale is unity, which is what the channeliser's unit-DC-gain prototype guarantees:
    a full-scale tone at a bin centre reads 0 dBFS.
    """
    clipped = np.maximum(power_linear, MINIMUM_LINEAR_POWER)
    return 10.0 * np.log10(clipped)


@dataclass(frozen=True)
class CalibrationPoint:
    """One measurement in a conducted sweep."""

    source_dbm: float
    attenuation_db: float
    measured_dbfs: float

    @property
    def input_dbm(self) -> float:
        """Power actually presented to the receiver input."""
        return self.source_dbm - self.attenuation_db


@dataclass
class GainCalibration:
    """A fitted dBFS-to-dBm relation for one gain configuration."""

    gains: HackrfGains
    offset_db: float
    slope: float
    r_squared: float
    residual_rms_db: float
    valid_dbfs_range: tuple[float, float]
    num_points: int

    def to_dbm(self, dbfs: float) -> float:
        """Convert a measured dBFS level to absolute input power."""
        return self.slope * dbfs + self.offset_db

    def in_range(self, dbfs: float) -> bool:
        """Whether ``dbfs`` falls inside the range the calibration was fitted over."""
        low, high = self.valid_dbfs_range
        return low <= dbfs <= high

    def as_dict(self) -> dict[str, Any]:
        return {
            "gains": self.gains.as_dict(),
            "offset_db": round(self.offset_db, 4),
            "slope": round(self.slope, 6),
            "r_squared": round(self.r_squared, 6),
            "residual_rms_db": round(self.residual_rms_db, 4),
            "valid_dbfs_range": [round(v, 3) for v in self.valid_dbfs_range],
            "num_points": self.num_points,
        }


def fit_calibration(
    gains: HackrfGains,
    points: list[CalibrationPoint],
    max_residual_db: float = 1.0,
) -> GainCalibration:
    """Fit dBm against dBFS over a conducted sweep, discarding non-linear points.

    Fits by least squares, then iteratively drops the worst-fitting point while any
    residual exceeds ``max_residual_db``. This is what isolates the linear region: the
    points that get dropped are the compressed ones at the top and the noise-limited ones
    at the bottom, and the range that survives becomes the calibration's stated validity.

    The fitted slope is a free parameter rather than being forced to 1.0 precisely so that
    it can be checked. A linear receiver must give one dB out per dB in; a slope that comes
    back at 0.8 means the sweep was taken through compression, or the attenuator is not
    what it says it is.

    Raises:
        ValueError: If fewer than three points remain, which is too few to distinguish a
            fit from a coincidence.
    """
    if len(points) < 3:
        raise ValueError(f"need at least 3 calibration points, got {len(points)}")

    remaining = list(points)
    while True:
        dbfs = np.array([p.measured_dbfs for p in remaining])
        dbm = np.array([p.input_dbm for p in remaining])
        slope, offset = np.polyfit(dbfs, dbm, 1)
        residuals = dbm - (slope * dbfs + offset)

        worst = float(np.abs(residuals).max())
        if worst <= max_residual_db or len(remaining) <= 3:
            break
        remaining.pop(int(np.argmax(np.abs(residuals))))

    total_variance = float(np.sum((dbm - dbm.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / total_variance if total_variance > 0 else 0.0

    return GainCalibration(
        gains=gains,
        offset_db=float(offset),
        slope=float(slope),
        r_squared=r_squared,
        residual_rms_db=float(np.sqrt(np.mean(residuals**2))),
        valid_dbfs_range=(float(dbfs.min()), float(dbfs.max())),
        num_points=len(remaining),
    )


class PowerCalibration:
    """Collection of gain-keyed calibrations, loaded from and saved to YAML.

    An uncalibrated instance is a valid and expected state, not an error: the node runs and
    reports dBFS, and every estimate derived from power is flagged as uncalibrated all the
    way through to the CoT message. That flag is the whole point — it is what stops an
    uncalibrated range estimate from being read as a measured one.
    """

    def __init__(self, calibrations: dict[str, GainCalibration] | None = None) -> None:
        self._calibrations = calibrations or {}

    @staticmethod
    def _key(gains: HackrfGains) -> str:
        return f"lna{gains.lna_db:.0f}_vga{gains.vga_db:.0f}_amp{int(gains.amp_enabled)}"

    @property
    def is_calibrated(self) -> bool:
        return bool(self._calibrations)

    def add(self, calibration: GainCalibration) -> None:
        self._calibrations[self._key(calibration.gains)] = calibration

    def get(self, gains: HackrfGains) -> GainCalibration | None:
        """Return the calibration for this exact gain configuration, if one exists."""
        return self._calibrations.get(self._key(gains))

    def to_dbm(self, dbfs: float, gains: HackrfGains) -> float | None:
        """Convert dBFS to absolute dBm, or ``None`` when no valid calibration applies.

        Returns ``None`` rather than extrapolating when there is no calibration for this
        gain configuration, or when the level falls outside the range the calibration was
        fitted over. Extrapolating past the fitted range is exactly how a compressed
        reading becomes a confident overestimate of emitter power, and from there an
        underestimate of range.
        """
        calibration = self.get(gains)
        if calibration is None or not calibration.in_range(dbfs):
            return None
        return calibration.to_dbm(dbfs)

    def save(self, path: Path) -> None:
        payload = {
            "version": 1,
            "calibrations": {k: v.as_dict() for k, v in self._calibrations.items()},
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> PowerCalibration:
        """Load calibrations from YAML. A missing file yields an uncalibrated instance."""
        if not path.exists():
            return cls()
        payload = yaml.safe_load(path.read_text()) or {}
        calibrations = {}
        for key, entry in (payload.get("calibrations") or {}).items():
            gain_entry = entry["gains"]
            calibrations[key] = GainCalibration(
                gains=quantise_gains(
                    gain_entry["lna_db"], gain_entry["vga_db"], gain_entry.get("amp_enabled", False)
                ),
                offset_db=entry["offset_db"],
                slope=entry["slope"],
                r_squared=entry["r_squared"],
                residual_rms_db=entry["residual_rms_db"],
                valid_dbfs_range=tuple(entry["valid_dbfs_range"]),
                num_points=entry["num_points"],
            )
        return cls(calibrations)


def estimate_dynamic_range_db(points: list[CalibrationPoint], linearity_db: float = 1.0) -> float:
    """Usable dynamic range implied by a sweep: the span that stays linear.

    Reported alongside the calibration because it bounds what the node can measure at all.
    An emitter above the top of this range reads compressed, one below the bottom is lost
    in the receiver's own noise, and in both cases the resulting power figure is wrong in a
    direction the system cannot detect from the sample alone.
    """
    if len(points) < 3:
        return 0.0
    fitted = fit_calibration(quantise_gains(0, 0), points, max_residual_db=linearity_db)
    low, high = fitted.valid_dbfs_range
    return high - low


def required_sweep_levels(
    max_input_dbm: float, dynamic_range_db: float = 60.0, step_db: float = 5.0
) -> list[float]:
    """Suggested input levels for a conducted sweep, strongest first.

    A 5 dB step over 60 dB gives thirteen points, which is enough to see curvature at both
    ends rather than merely fitting a line through the middle and assuming the rest.
    """
    count = int(math.floor(dynamic_range_db / step_db)) + 1
    return [max_input_dbm - i * step_db for i in range(count)]
