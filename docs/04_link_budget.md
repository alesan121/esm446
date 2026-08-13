# Link budget and measured receiver performance

What the receiver hears, what it was predicted to hear, and where the two have and have not
been compared. Everything predicted here is computed by `matlab/link_budget.m` and by
`esm446/core/rfchain.py`, which are checked against each other by
`tests/test_matlab_consistency.py`; everything measured was taken on the hardware described
below.

## Configuration

| | |
|---|---|
| Receiver | HackRF One, direct conversion, 8-bit converter |
| Antenna | Wideband telescopic, 40 MHz – 6 GHz, cut to a quarter wave (16.8 cm) |
| External LNA | 20 dB, **modelled but not fitted for any measurement below** |
| Centre frequency | 446.593 75 MHz, offset-tuned — see `esm446/core/bands.py` |
| Sample rate | 2 MS/s, 160 channels of 12.5 kHz |

## Predicted sensitivity

Cascaded noise figure by Friis, in a 12.5 kHz channel bandwidth:

| | without LNA | with 20 dB LNA |
|---|---|---|
| cascaded noise figure | 8.50 dB | **1.69 dB** |
| noise floor | −124.5 dBm | −131.3 dBm |
| minimum detectable signal at 13 dB SNR | −111.5 dBm | **−118.3 dBm** |

The first stage sets the noise figure, so 20 dB of gain at the antenna divides the receiver's
own 8 dB contribution by a hundred. That is 6.8 dB, which under a log-distance exponent of
3.5 is about 57 % more detection range.

**None of this has been verified against a measurement**, and it cannot be until there is a
power calibration. See [#41](https://github.com/alesan121/esm446/issues/41): absolute power
requires a conducted calibration with an attenuator, which is not available, so every
`estimated_dbm` the node produces is `null`. The figures above are computed from
datasheet-typical values for a device that has not been characterised.

## Measured: is the receiver limited by its own noise or the environment?

Antenna disconnected against antenna connected, both at LNA 0 dB / VGA 0 dB, no transmission:

| | noise floor |
|---|---|
| antenna disconnected | −79.32 dBFS |
| antenna connected | −78.81 dBFS |
| **difference** | **0.51 dB** |

External noise therefore sits about 9 dB below the receiver's own. The system is
internal-noise-limited, which is the condition under which an external LNA's noise figure
improvement is realised almost in full.

**This measurement overstates its own conclusion.** It was taken at LNA 0 / VGA 0, the gain
setting used to avoid saturating with a transmitter three metres away, and the HackRF's noise
figure is considerably worse at low gain than at its normal operating point. Repeating the
pair at LNA 32 / VGA 20 would settle it. Until that is done, the honest statement is that the
receiver is internal-noise-limited *at low gain*, and the LNA is likely but not proven to be
worth its 6.8 dB.

## Measured: the receiver's own artefacts

A direct-conversion receiver puts a spur at its own centre frequency, from local-oscillator
leakage into the mixer. Measured at **+31 dB above the noise floor**, and confirmed as an
artefact rather than a signal by retuning — the peak followed the local oscillator to the
hertz:

| local oscillator | peak | offset | level |
|---|---|---|---|
| 446.093 750 MHz | 446.093 750 MHz | **+0 Hz** | −41.9 dBFS |
| 446.593 750 MHz | 446.593 750 MHz | **+0 Hz** | −41.9 dBFS |

IQ imbalance additionally mirrors every signal about DC. With the centre inside the
allocation, those mirrors land on other channels — channel 9 onto channel 7, 10 onto 6, 12
onto 4 — so one real emitter raises a phantom on its mirror. Offset tuning places the spur
400 kHz above channel 16 and every image above the band; `matlab/channel_plan.m` computes
both.

## Measured: near-field test geometry

The only space available for a controlled transmission test is three metres indoors. At
446 MHz the free-space loss over that distance is 35 dB, so a handset at minimum power arrives
at the receiver's linear limit exactly:

| | received | verdict |
|---|---|---|
| handset at 1 W, antennas aligned | −5.0 dBm | at the linear limit, compressing |
| handset at 1 W, antennas cross-polarised | **−25.0 dBm** | 20 dB of margin |
| handset at 5 W, antennas aligned | **+2.0 dBm** | 7 dB into compression |

Distance is the obvious lever and indoors it is the one unavailable. Turning the antennas at
right angles is worth more than tripling the separation and costs nothing.
`matlab/measurement_setup.m` computes the requirement for any power and separation, and
floors its answers at three wavelengths because free-space loss describes nothing real closer
than that.

Note that cross-polarisation isolation was assumed at 20 dB and, in practice at three metres,
proved insufficient on its own: the first transmission test clipped the converter and had to
be repeated with the receiver's internal gains at zero.

## Measured: transmitter spectral purity

Both handsets show a **discrete spurious pair at ±37.5 kHz** from the carrier. The profile is
not monotonic, which is what distinguishes a spur from modulation splatter:

| offset | radio 1 | radio 2 |
|---|---|---|
| ±12.5 kHz | — | −44.8 / −48.4 dBc |
| ±25.0 kHz | −45.4 dBc | −45.8 / −46.4 dBc |
| **±37.5 kHz** | **−32.8 to −34.6 dBc** | **−34.9 / −35.2 dBc** |
| ±50.0 kHz | −45.1 dBc | −45.7 / −46.0 dBc |

The ratio is independent of transmit power — −34.6 dBc at high, −31.9 dBc at low — so it is a
fixed-ratio synthesiser artefact rather than a power-dependent nonlinearity in the amplifier.
37.5 kHz is three channel steps, which points at the fractional-N synthesiser.

Both units of the same model show it within about 2 dB. That is the shape of a **specific
emitter identification** feature: it distinguishes two radios that share a channel plan and,
in the test that produced these numbers, would have shared a CTCSS code.

The node reports these as detections on the neighbouring channels, because they are energy on
those channels and the detector is right about them. `esm446.analysis.artefacts` then
attributes each to the carrier that produced it, by the arithmetic relation it obeys — the
splatter pairs are symmetric about the carrier to within 211 Hz. They are kept rather than
suppressed, since hiding them would discard this measurement.

## Measured: transmitter intermodulation

With both handsets transmitting at once, 62.5 kHz apart, a second mechanism appears that a
single transmitter cannot produce — third-order products at 2·f₁−f₂ and 2·f₂−f₁:

| predicted | measured | error | level |
|---|---|---|---|
| 445.968 813 MHz | 445.968 917 MHz | **+104 Hz** | −22.0 dBc |
| 446.156 190 MHz | 446.156 089 MHz | **−101 Hz** | −25.1 dBc |

Each product carries the sub-audible tone of the carrier whose frequency is doubled, which is
what the mixing implies and confirms the attribution independently of the frequencies.

Where the nonlinearity sits is not established. Two handsets three metres apart each receive
the other at a substantial level, so transmitter intermodulation is as plausible as the
receiver's own; separating them needs the products measured against transmitter separation,
which the available space does not allow. The receiver was at LNA 0 / VGA 0 and not
compressing on either carrier, which makes the receiver the less likely of the two but does
not settle it.

## What remains unmeasured

- **Absolute power.** Blocked on [#41](https://github.com/alesan121/esm446/issues/41).
- **Antenna gain.** The link budget uses a conservative default rather than a supplier figure,
  because those figures are not physical at this frequency — see `esm446/core/antenna.py`.
  Gain by substitution against a quarter-wave reference needs no extra equipment and has not
  been done.
- **Frequency accuracy.** Reported frequencies are consistent to within tens of hertz across
  captures, but nothing has established that they are *correct*. The HackRF's crystal drifts
  by several parts per million, which at 446 MHz is hundreds of hertz of systematic error.
  Calibrating against a broadcast carrier would settle it.
- **Achieved image rejection.** Where the images fall is computed; how far down they are is
  not measured.
