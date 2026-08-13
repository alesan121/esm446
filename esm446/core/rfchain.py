"""Receiver chain model: noise figure, sensitivity, gain planning and damage limits.

This module is the link budget as executable code rather than a spreadsheet, for one
reason: the predicted sensitivity has to be comparable against what the node actually
measures. A number in a document drifts away from the system; a number computed from the
same configuration the node runs cannot.

The chain as deployed
---------------------
An external 20 dB LNA ahead of a HackRF One. That ordering is the whole point, and Friis's
formula says why::

    F_total = F1 + (F2 - 1)/G1

The HackRF's own noise figure is around 8 dB. Placed behind a 20 dB LNA of roughly 1 dB
noise figure, its contribution is divided by 100, and the system noise figure collapses
from about 8 dB to about 1.2 dB. That is close to 7 dB of free sensitivity — in range
terms, under a log-distance exponent of 3.5, roughly a 56 % increase in detection radius.
The first stage sets the noise figure; everything after it is nearly irrelevant to noise
and entirely relevant to dynamic range.

Note the units trap in that calculation: noise *factors* cascade additively, noise
*figures* in dB do not. Adding 1.2589 and 0.0531 gives a factor of 1.3120, whose dB value
is 1.18 — not 1.31. Mixing the two produces answers that look reasonable and are wrong.

Dynamic range is the other side of that trade. The HackRF digitises with 8 bits, so the
usable window between the noise floor and clipping is around 48 dB. Adding 20 dB of gain in
front means the internal gains must come *down* by roughly the same amount, or strong local
signals drive the ADC into clipping and every power measurement downstream becomes fiction.
`plan_gains` does that arithmetic.

Damage limits
-------------
An external LNA is the most fragile component in the chain and the easiest to destroy. Two
specific ways to do it, both reachable during the conducted calibration this project calls
for:

- Feeding a signal generator into it without enough attenuation. Survival input is around
  +10 dBm and compression starts far below that.
- Transmitting into it. The HackRF and the PortaPack both transmit. An LNA on the antenna
  port during transmit is destroyed instantly.

`check_input_safety` exists so the calibration tooling refuses to proceed rather than
relying on the operator remembering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Thermal noise power spectral density at 290 K, in dBm/Hz.
THERMAL_NOISE_DBM_PER_HZ = -174.0

#: HackRF One internal gain stages and their quantisation, from the hackrf documentation.
#: Requesting an unquantised value silently snaps to the nearest step in the driver, so the
#: node snaps explicitly and records what it actually asked for.
HACKRF_LNA_STEP_DB = 8.0
HACKRF_LNA_MAX_DB = 40.0
HACKRF_VGA_STEP_DB = 2.0
HACKRF_VGA_MAX_DB = 62.0
HACKRF_AMP_GAIN_DB = 14.0

#: Recommended maximum input to the HackRF antenna port for linear operation (dBm).
#: Damage threshold is higher, around +10 dBm, but compression well below that makes any
#: power measurement meaningless.
HACKRF_MAX_LINEAR_INPUT_DBM = -5.0


@dataclass(frozen=True)
class Stage:
    """One element of the receive chain, in signal order.

    Attributes:
        name: Identifier used in reports.
        gain_db: Gain in dB. Negative for cable and attenuator losses.
        noise_figure_db: Noise figure in dB. For a passive lossy element this equals the
            loss, which is why cable ahead of the LNA costs sensitivity directly.
        max_input_dbm: Input level above which this stage stops being usable. This is the
            *linear* limit, not the survival limit: an amplifier driven into compression
            has not been destroyed, but every power measurement taken through it is wrong,
            which for a system whose output is calibrated power is the same thing.
    """

    name: str
    gain_db: float
    noise_figure_db: float
    max_input_dbm: float = 10.0


@dataclass
class RfChain:
    """A cascade of receive stages, ordered from antenna inward."""

    stages: list[Stage] = field(default_factory=list)

    @classmethod
    def deployed(
        cls,
        lna_gain_db: float = 20.0,
        lna_noise_figure_db: float = 1.0,
        cable_loss_db: float = 0.5,
        hackrf_noise_figure_db: float = 8.0,
    ) -> RfChain:
        """The chain as fielded: antenna cable, external LNA, short cable, HackRF.

        Defaults are typical rather than measured. Replace ``lna_noise_figure_db`` and
        ``lna_gain_db`` with the datasheet or measured values for the actual device before
        citing any sensitivity figure as fact.
        """
        return cls(
            [
                Stage("antenna cable", -cable_loss_db, cable_loss_db, max_input_dbm=30.0),
                # -10 dBm is the linear limit, not the survival limit. A 20 dB LNA with a
                # +18 dBm output P1dB compresses from about -2 dBm at its input; survival
                # is nearer +10 dBm. Compression is the binding constraint because it
                # corrupts measurements long before anything smells of smoke.
                Stage("external LNA", lna_gain_db, lna_noise_figure_db, max_input_dbm=-10.0),
                Stage("LNA-to-SDR cable", -0.2, 0.2, max_input_dbm=30.0),
                Stage(
                    "HackRF One",
                    0.0,
                    hackrf_noise_figure_db,
                    max_input_dbm=HACKRF_MAX_LINEAR_INPUT_DBM,
                ),
            ]
        )

    @property
    def total_gain_db(self) -> float:
        """Total gain of the chain, excluding the HackRF's own configurable gains."""
        return sum(stage.gain_db for stage in self.stages)

    @property
    def noise_figure_db(self) -> float:
        """Cascaded noise figure by Friis's formula.

        Computed in linear noise factor, because noise factors cascade additively while
        noise figures in dB do not — a mistake that produces plausible-looking numbers.
        """
        total_factor = 0.0
        gain_so_far = 1.0
        for stage in self.stages:
            factor = 10.0 ** (stage.noise_figure_db / 10.0)
            total_factor += (factor - 1.0) / gain_so_far
            gain_so_far *= 10.0 ** (stage.gain_db / 10.0)
        return float(10.0 * np.log10(1.0 + total_factor))

    def noise_floor_dbm(self, bandwidth_hz: float) -> float:
        """Thermal noise power referred to the antenna, in the given bandwidth."""
        return THERMAL_NOISE_DBM_PER_HZ + 10.0 * np.log10(bandwidth_hz) + self.noise_figure_db

    def minimum_detectable_signal_dbm(
        self, bandwidth_hz: float, required_snr_db: float = 13.0
    ) -> float:
        """Weakest signal detectable at the given SNR, referred to the antenna.

        The default ``required_snr_db`` of 13 dB corresponds to single-frame detection at
        the CFAR design point of P_fa = 1e-4 with a useful probability of detection.
        Integrating over multiple frames lowers it; the V&V report measures the achieved
        figure against this prediction rather than assuming they agree.
        """
        return self.noise_floor_dbm(bandwidth_hz) + required_snr_db

    def check_input_safety(self, source_power_dbm: float) -> list[str]:
        """Return a list of stage-level violations for a given input power.

        Applied before any conducted measurement. The check walks the chain accumulating
        gain, so it catches the case that actually destroys hardware: a level that is
        harmless at the antenna port becoming damaging once the LNA has amplified it.

        Returns an empty list when the chain is safe.
        """
        violations = []
        level = source_power_dbm
        for stage in self.stages:
            if level > stage.max_input_dbm:
                violations.append(
                    f"{stage.name}: {level:.1f} dBm at input exceeds its "
                    f"{stage.max_input_dbm:.1f} dBm limit"
                )
            level += stage.gain_db
        return violations

    def required_attenuation_db(self, source_power_dbm: float, margin_db: float = 10.0) -> float:
        """Attenuation needed ahead of the chain for a conducted measurement.

        Args:
            source_power_dbm: Output level of the signal generator.
            margin_db: Headroom below the tightest limit in the chain.

        Returns:
            Attenuation in dB, rounded up to the next 10 dB step since that is how
            attenuator pads come. Zero if none is needed.
        """
        needed = 0.0
        level = source_power_dbm
        for stage in self.stages:
            excess = level - (stage.max_input_dbm - margin_db)
            needed = max(needed, excess)
            level += stage.gain_db
        return float(max(0.0, np.ceil(needed / 10.0) * 10.0))

    def describe(self, bandwidth_hz: float = 12_500.0) -> str:
        """Human-readable link budget summary for reports."""
        lines = [f"{'stage':<20} {'gain dB':>8} {'NF dB':>7}"]
        for stage in self.stages:
            lines.append(f"{stage.name:<20} {stage.gain_db:>8.1f} {stage.noise_figure_db:>7.1f}")
        lines.append("")
        lines.append(f"cascaded noise figure   {self.noise_figure_db:>6.2f} dB")
        lines.append(
            f"noise floor @ {bandwidth_hz / 1e3:.1f} kHz  {self.noise_floor_dbm(bandwidth_hz):>6.1f} dBm"
        )
        lines.append(
            f"MDS @ 13 dB SNR         {self.minimum_detectable_signal_dbm(bandwidth_hz):>6.1f} dBm"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class HackrfGains:
    """A quantised HackRF gain setting.

    Recorded with every detection. The power-to-dBm calibration is only valid for the exact
    gain configuration it was measured at, so a stored power reading without its gains is
    not calibratable after the fact — which is the flaw that made v0's OFFSET_CAL
    unreproducible.
    """

    lna_db: float
    vga_db: float
    amp_enabled: bool = False

    @property
    def total_db(self) -> float:
        return self.lna_db + self.vga_db + (HACKRF_AMP_GAIN_DB if self.amp_enabled else 0.0)

    def as_dict(self) -> dict[str, float | bool]:
        return {"lna_db": self.lna_db, "vga_db": self.vga_db, "amp_enabled": self.amp_enabled}


def quantise_gains(lna_db: float, vga_db: float, amp_enabled: bool = False) -> HackrfGains:
    """Snap requested gains to the steps the HackRF can actually apply.

    The driver rounds silently. Doing it here means the value recorded in the metadata is
    the value the hardware used, not the value someone asked for.
    """
    lna = float(
        np.clip(round(lna_db / HACKRF_LNA_STEP_DB) * HACKRF_LNA_STEP_DB, 0.0, HACKRF_LNA_MAX_DB)
    )
    vga = float(
        np.clip(round(vga_db / HACKRF_VGA_STEP_DB) * HACKRF_VGA_STEP_DB, 0.0, HACKRF_VGA_MAX_DB)
    )
    return HackrfGains(lna_db=lna, vga_db=vga, amp_enabled=amp_enabled)


def plan_gains(external_gain_db: float, target_total_gain_db: float = 52.0) -> HackrfGains:
    """Choose HackRF internal gains given whatever gain is already ahead of it.

    With an 8-bit ADC there is about 48 dB between the noise floor and clipping, so total
    chain gain is a budget, not a free parameter. Every dB added externally must come out
    of the internal stages or the converter clips on strong local signals and every power
    measurement downstream becomes fiction.

    LNA is filled before VGA because it sits ahead of the mixer, where gain still improves
    the internal noise figure; baseband VGA gain amplifies noise and signal alike.
    """
    budget = max(0.0, target_total_gain_db - external_gain_db)
    lna = min(HACKRF_LNA_MAX_DB, np.floor(budget / HACKRF_LNA_STEP_DB) * HACKRF_LNA_STEP_DB)
    vga = min(HACKRF_VGA_MAX_DB, budget - lna)
    return quantise_gains(lna, vga, amp_enabled=False)
