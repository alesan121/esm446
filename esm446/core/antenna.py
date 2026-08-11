"""Antenna catalogue, band coverage, and a physical plausibility check on claimed gain.

Why this module exists
----------------------
Antenna gain enters the link budget directly: every dB of it is a dB of predicted
detection range. So a wrong gain figure does not produce a slightly wrong answer, it
produces a confident one. The gain figures printed on consumer antenna listings are
frequently not physical, and feeding them into a link budget is how a system ends up
predicting a detection range ten times what it achieves.

The check is elementary and hard to argue with. The gain of any aperture antenna is bounded
by its physical size::

    G = 4 * pi * A_eff / lambda**2,     A_eff = efficiency * A_physical

At 446 MHz the wavelength is 0.67 m. A genuine 35 dBi — a factor of 3162 — would need an
effective aperture of about 114 m², which is a dish roughly 12 m across. An antenna that
fits in a rucksack cannot have it, at any price, from any manufacturer. This is not a
quality judgement about a particular product; it is a statement about apertures.

`max_plausible_gain_dbi` implements the bound so the link budget refuses inflated figures
instead of quietly propagating them.

Coverage matters more than gain here
------------------------------------
PMR446 sits at 446 MHz. An antenna specified from 700 MHz upward is not a slightly
suboptimal choice at 446 MHz — it is below its lower band edge, where the feedpoint
impedance has left the design region entirely. Most of the incident power is reflected
rather than delivered, so the effective gain is strongly negative regardless of what the
label claims. `covers` is checked before gain is even considered.

Obtaining a real number
-----------------------
The honest way to get antenna gain without an anechoic chamber is **gain by substitution**:
receive the same source, from the same position, through each antenna in turn, and compare
received power. That yields relative gain directly, and absolute gain if one antenna in the
set has a known figure — a quarter-wave whip over a decent ground plane is close enough to
2.15 dBi to serve as the reference. It needs no equipment beyond what this project already
uses, and it turns a marketing number into a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0

#: Aperture efficiency assumed when bounding plausible gain. 0.6 is generous for anything
#: that is not a well-fed parabolic reflector, so the bound errs towards accepting claims.
APERTURE_EFFICIENCY = 0.6

#: Gain of a half-wave dipole over an isotropic radiator (dBi). The practical reference for
#: substitution measurements.
DIPOLE_GAIN_DBI = 2.15


def wavelength(frequency_hz: float) -> float:
    """Free-space wavelength in metres."""
    return SPEED_OF_LIGHT / frequency_hz


def max_plausible_gain_dbi(frequency_hz: float, largest_dimension_m: float) -> float:
    """Upper bound on the gain of an antenna of a given physical size.

    Uses the aperture relation with the antenna's largest dimension squared as a generous
    stand-in for physical aperture. Real antennas fall below this; nothing exceeds it by
    a meaningful margin without becoming a superdirective array, which is narrowband,
    lossy, and not what is being sold on a marketplace listing.
    """
    aperture = APERTURE_EFFICIENCY * largest_dimension_m**2
    gain_linear = 4.0 * np.pi * aperture / wavelength(frequency_hz) ** 2
    return float(10.0 * np.log10(max(gain_linear, 1e-6)))


def required_dimension_m(frequency_hz: float, gain_dbi: float) -> float:
    """Largest dimension an antenna would need to achieve ``gain_dbi``.

    The number that makes an implausible claim obvious at a glance.
    """
    gain_linear = 10.0 ** (gain_dbi / 10.0)
    aperture = gain_linear * wavelength(frequency_hz) ** 2 / (4.0 * np.pi)
    return float(np.sqrt(aperture / APERTURE_EFFICIENCY))


@dataclass(frozen=True)
class Antenna:
    """An antenna, with its claimed specification kept separate from anything measured.

    Attributes:
        name: Identifier used in reports.
        min_frequency_hz: Lower edge of the specified band.
        max_frequency_hz: Upper edge of the specified band.
        claimed_gain_dbi: Gain as specified by the supplier. Never used in the link budget.
        largest_dimension_m: Longest physical dimension, used for the plausibility bound.
        measured_gain_dbi: Gain obtained by substitution measurement, if one has been done.
            This is the only figure the link budget will use.
    """

    name: str
    min_frequency_hz: float
    max_frequency_hz: float
    claimed_gain_dbi: float | None = None
    largest_dimension_m: float | None = None
    measured_gain_dbi: float | None = None

    def covers(self, frequency_hz: float) -> bool:
        """Whether the antenna is specified to operate at this frequency."""
        return self.min_frequency_hz <= frequency_hz <= self.max_frequency_hz

    def plausibility(self, frequency_hz: float) -> str | None:
        """Return a description of why the claimed gain is not physical, or ``None``.

        Returns ``None`` when there is nothing to object to, including when no gain is
        claimed or no dimension is known.
        """
        if self.claimed_gain_dbi is None or self.largest_dimension_m is None:
            return None
        bound = max_plausible_gain_dbi(frequency_hz, self.largest_dimension_m)
        if self.claimed_gain_dbi <= bound + 1.0:
            return None
        needed = required_dimension_m(frequency_hz, self.claimed_gain_dbi)
        return (
            f"{self.name}: {self.claimed_gain_dbi:.0f} dBi claimed at "
            f"{frequency_hz / 1e6:.1f} MHz would need an aperture roughly {needed:.1f} m "
            f"across; this antenna is {self.largest_dimension_m:.2f} m, bounding it at "
            f"{bound:.1f} dBi"
        )

    def effective_gain_dbi(self, frequency_hz: float, fallback_dbi: float = 0.0) -> float:
        """Gain to use in a link budget.

        Returns the measured gain when one exists. Otherwise returns ``fallback_dbi`` — a
        deliberately conservative default — rather than the claimed figure. An unmeasured
        antenna contributes an assumption to the budget, and the budget should say so.
        """
        if self.measured_gain_dbi is not None:
            return self.measured_gain_dbi
        return fallback_dbi


#: The antennas available for this project, as specified by their suppliers.
#:
#: Only two of the five are specified to cover 446 MHz. The 700-2700 MHz and 2.4/5.8 GHz
#: units are not marginal choices at 446 MHz, they are out of band. Dimensions are nominal
#: and should be replaced with measured ones.
CATALOGUE: tuple[Antenna, ...] = (
    Antenna(
        "wideband 40-860 MHz",
        40e6,
        860e6,
        claimed_gain_dbi=None,
        largest_dimension_m=1.0,
    ),
    Antenna(
        "wideband 40 MHz - 6 GHz",
        40e6,
        6e9,
        claimed_gain_dbi=None,
        largest_dimension_m=0.5,
    ),
    Antenna(
        "panel 700-2700 MHz 35 dBi",
        700e6,
        2700e6,
        claimed_gain_dbi=35.0,
        largest_dimension_m=0.3,
    ),
    Antenna(
        "panel 700-2700 MHz 12 dBi",
        700e6,
        2700e6,
        claimed_gain_dbi=12.0,
        largest_dimension_m=0.25,
    ),
    Antenna(
        "8 dBi 2.4/5/5.8 GHz",
        2400e6,
        5900e6,
        claimed_gain_dbi=8.0,
        largest_dimension_m=0.2,
    ),
)


def usable_at(frequency_hz: float, catalogue: tuple[Antenna, ...] = CATALOGUE) -> list[Antenna]:
    """Antennas from the catalogue specified to cover ``frequency_hz``."""
    return [antenna for antenna in catalogue if antenna.covers(frequency_hz)]


def audit(frequency_hz: float, catalogue: tuple[Antenna, ...] = CATALOGUE) -> list[str]:
    """Report coverage and plausibility problems across a catalogue at one frequency."""
    findings = []
    for antenna in catalogue:
        if not antenna.covers(frequency_hz):
            findings.append(
                f"{antenna.name}: does not cover {frequency_hz / 1e6:.3f} MHz "
                f"(specified {antenna.min_frequency_hz / 1e6:.0f}-"
                f"{antenna.max_frequency_hz / 1e6:.0f} MHz)"
            )
            continue
        problem = antenna.plausibility(frequency_hz)
        if problem:
            findings.append(problem)
    return findings
