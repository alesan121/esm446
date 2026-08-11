"""PMR446 band plan and the receiver tuning that makes it align with the channeliser.

The analogue PMR446 allocation (ETSI EN 300 296) is 16 channels of 12.5 kHz spacing,
channel 1 centred at 446.006 25 MHz and channel 16 at 446.193 75 MHz.

The non-obvious part of this module is `DEFAULT_CENTRE_HZ`. A polyphase channeliser
produces bins at ``centre + k * sample_rate / num_channels``. For those bins to land
*exactly* on the PMR446 channel centres — no half-bin offset, no resampling downstream —
the receiver centre frequency must sit an integer number of 12.5 kHz steps away from the
channel grid origin. Channel 8 satisfies this by construction:

    446.093 75 MHz - 446.006 25 MHz = 87.5 kHz = 7 x 12.5 kHz

The midpoint of the allocation (446.1 MHz) does *not*: it is offset by 7.5 steps, i.e. half
a bin, which would smear every channel across two bins. The v0 prototype used
446.093 5 MHz, 250 Hz away from channel 8, which introduced a small but entirely avoidable
tuning error.
"""

from __future__ import annotations

import numpy as np

#: Centre frequency of PMR446 channel 1 (Hz).
CHANNEL_1_HZ = 446_006_250

#: Channel spacing of the analogue PMR446 allocation (Hz).
CHANNEL_SPACING_HZ = 12_500

#: Number of analogue PMR446 channels.
CHANNEL_COUNT = 16

#: Receiver centre frequency (Hz) — PMR446 channel 8. See module docstring.
DEFAULT_CENTRE_HZ = 446_093_750

#: Nominal occupied bandwidth of a 12.5 kHz NFM PMR446 emission (Hz).
#: ETSI EN 300 296 permits max 2.5 kHz deviation; Carson's rule with 3 kHz audio gives
#: 2 * (2.5 + 3.0) = 11 kHz.
OCCUPIED_BANDWIDTH_HZ = 11_000


def channel_frequency(channel: int) -> int:
    """Centre frequency in Hz of PMR446 analogue channel ``channel`` (1-16)."""
    if not 1 <= channel <= CHANNEL_COUNT:
        raise ValueError(f"PMR446 channel must be 1-{CHANNEL_COUNT}, got {channel}")
    return CHANNEL_1_HZ + (channel - 1) * CHANNEL_SPACING_HZ


#: How far an emission may sit from a nominal channel centre and still be called that channel.
#:
#: Nothing obliges a transmitter to sit on the grid. Inexpensive handsets are routinely a few
#: hundred hertz off nominal, and an emitter may be anywhere in the band by accident or by
#: intent — which is a large part of why the whole 2 MHz is surveyed rather than only the 16
#: nominal channels.
#:
#: So the tolerance has to be wide enough to absorb ordinary crystal error and narrow enough
#: that a genuinely off-grid emitter is reported as off-grid rather than snapped to its
#: nearest neighbour. 2 kHz is a sixth of the channel spacing: comfortably beyond handset
#: drift, comfortably inside the 6.25 kHz that separates a channel centre from the midpoint
#: between two.
CHANNEL_MATCH_TOLERANCE_HZ = 2_000.0


def channel_at(frequency_hz: float, tolerance_hz: float = CHANNEL_MATCH_TOLERANCE_HZ) -> int | None:
    """Return the PMR446 channel number at ``frequency_hz``, or ``None`` if off-grid.

    Off-grid emissions are not errors — surveying the band beyond the nominal 16 channels
    is a deliberate capability — so this returns ``None`` rather than raising, and callers
    record the absolute frequency instead.
    """
    for channel in range(1, CHANNEL_COUNT + 1):
        if abs(frequency_hz - channel_frequency(channel)) <= tolerance_hz:
            return channel
    return None


def bin_frequencies(centre_hz: float, sample_rate: float, num_channels: int) -> np.ndarray:
    """Absolute centre frequency (Hz) of each channeliser bin, in FFT bin order.

    Bins ``0 .. M/2-1`` are above ``centre_hz``; bins ``M/2 .. M-1`` are below it, following
    the usual FFT convention.
    """
    offsets = np.fft.fftfreq(num_channels, d=1.0 / sample_rate)
    return centre_hz + offsets


def channel_bin_index(channel: int, centre_hz: float, sample_rate: float, num_channels: int) -> int:
    """Channeliser bin index carrying PMR446 ``channel``.

    Raises ``ValueError`` if the channel does not fall exactly on a bin centre, which means
    the centre frequency or the channel count has been chosen inconsistently — a
    configuration error worth failing loudly on rather than silently mistuning.
    """
    spacing = sample_rate / num_channels
    offset = channel_frequency(channel) - centre_hz
    steps = offset / spacing
    nearest = round(steps)
    if abs(steps - nearest) > 1e-9:
        raise ValueError(
            f"PMR446 channel {channel} is {steps:.4f} bins from the centre frequency, "
            f"not an integer. Centre {centre_hz / 1e6:.6f} MHz with {num_channels} channels "
            f"at {sample_rate / 1e6:.3f} MS/s does not align with the 12.5 kHz grid; "
            f"use bands.DEFAULT_CENTRE_HZ."
        )
    return int(nearest % num_channels)
