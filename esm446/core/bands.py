"""PMR446 band plan and the receiver tuning that makes it align with the channeliser.

The analogue PMR446 allocation (ETSI EN 300 296) is 16 channels of 12.5 kHz spacing,
channel 1 centred at 446.006 25 MHz and channel 16 at 446.193 75 MHz.

The non-obvious part of this module is `DEFAULT_CENTRE_HZ`, and it is set by two constraints
that pull in different directions.

**Grid alignment.** A polyphase channeliser produces bins at
``centre + k * sample_rate / num_channels``. For those bins to land *exactly* on the PMR446
channel centres — no half-bin offset, no resampling downstream — the receiver centre must sit
an integer number of 12.5 kHz steps from the channel grid origin. The midpoint of the
allocation, 446.1 MHz, fails this: it is 7.5 steps out, half a bin, which would smear every
channel across two. The v0 prototype used 446.093 5 MHz, 250 Hz from channel 8, a small and
entirely avoidable tuning error.

**Keeping the receiver's own artefacts out of the band.** The HackRF is a direct-conversion
receiver, and that costs two things at the centre frequency:

- *LO leakage* appears as a DC term, so a spur sits permanently at the centre. Measured on
  hardware at **+31 dB above the noise floor**, and confirmed as an artefact rather than a
  signal by retuning: the peak followed the local oscillator to the hertz.
- *IQ imbalance* mirrors every signal about DC. With the centre inside the allocation those
  mirrors land on real channels — a signal on channel 9 raises a phantom on channel 7.

Choosing channel 8 satisfied the first constraint and violated the second as badly as
possible: the spur landed on a nominal channel, and channel pairs mirrored onto each other.

The answer is **offset tuning**: stay on the 12.5 kHz grid, but place the centre outside the
allocation. 446.593 75 MHz is 47 steps from channel 1, so every channel still lands exactly
on a bin, while the DC spur sits 400 kHz above channel 16 and every image falls above the
band. `assert_centre_is_usable` refuses a centre frequency that coincides with a channel,
because that guarantees the spur sits on one.
"""

from __future__ import annotations

import numpy as np

#: Centre frequency of PMR446 channel 1 (Hz).
CHANNEL_1_HZ = 446_006_250

#: Channel spacing of the analogue PMR446 allocation (Hz).
CHANNEL_SPACING_HZ = 12_500

#: Number of analogue PMR446 channels.
CHANNEL_COUNT = 16

#: Receiver centre frequency (Hz). Offset-tuned: on the 12.5 kHz grid, 47 steps from channel
#: 1, but 400 kHz above channel 16 so the DC spur and the IQ images fall outside the
#: allocation. See the module docstring for the measurements behind it.
DEFAULT_CENTRE_HZ = 446_593_750

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


def assert_centre_is_usable(centre_hz: float) -> None:
    """Raise if a centre frequency would put the receiver's DC spur on a nominal channel.

    A direct-conversion receiver always leaks its local oscillator into the mixer, producing
    a spur at exactly the centre frequency. Measured on a HackRF One that spur is about 31 dB
    above the noise floor, which is far stronger than most real emissions in this band, so a
    centre frequency sitting on a channel guarantees a permanent phantom emitter there.

    Args:
        centre_hz: Proposed receiver centre frequency.

    Raises:
        ValueError: If the frequency coincides with a PMR446 channel.
    """
    channel = channel_at(centre_hz, tolerance_hz=1.0)
    if channel is not None:
        raise ValueError(
            f"centre frequency {centre_hz / 1e6:.6f} MHz is PMR446 channel {channel}. A "
            f"direct-conversion receiver puts a DC spur at its own centre frequency, so this "
            f"would report a permanent phantom emitter on channel {channel}. Offset-tune "
            f"instead: use {DEFAULT_CENTRE_HZ} Hz, which stays on the 12.5 kHz grid but "
            f"places the spur outside the allocation."
        )


def image_frequency(signal_hz: float, centre_hz: float) -> float:
    """Frequency at which IQ imbalance mirrors ``signal_hz`` about the centre.

    Args:
        signal_hz: Frequency of the real signal.
        centre_hz: Receiver centre frequency.

    Returns:
        The mirrored frequency.
    """
    return 2.0 * centre_hz - signal_hz


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
