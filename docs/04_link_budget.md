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
not monotonic, which is what distinguishes a spur from modulation splatter.

The two radios are **different models from different manufacturers**: a Baofeng UV-5RA and a
Radtel RT-900. Measured per radio, averaged over the frames each carrier was up, relative to
its own carrier:

| offset | UV-5RA (alone) | UV-5RA (both on air) | RT-900 (both on air) |
|---|---|---|---|
| ±12.5 kHz | −44.3 / −44.2 | −38.1 / −38.2 | −39.4 / −39.2 |
| ±25.0 kHz | −42.6 / −43.3 | −34.3 / −37.8 | −38.7 / −35.4 |
| **±37.5 kHz** | **−34.7 / −34.7** | **−33.5 / −33.4** | **−33.7 / −33.7** |
| ±50.0 kHz | −41.5 / −41.5 | −37.8 / −37.8 | −38.6 / −38.7 |

The single-radio column is the clean one. With both on air the carriers are five channel steps
apart, so each radio's ±25 and ±50 kHz bins carry the other's energy and their intermodulation
products; only the ±37.5 kHz figures are uncontaminated in that capture.

The ratio is independent of transmit power — −34.6 dBc at high, −31.9 dBc at low — so it is a
fixed-ratio synthesiser artefact rather than a power-dependent nonlinearity in the amplifier.
37.5 kHz is three channel steps, which points at the fractional-N synthesiser.

### It is not a specific-emitter-identification feature

An earlier version of this document drew the opposite conclusion, on the belief that the two
radios were two units of one model. They are not, and that changes what the measurement means.

Across two different models from two manufacturers the spur sits at **−33.4 to −34.7 dBc**, a
spread of about 1.2 dB — which is within the uncertainty of these measurements. A feature that
takes the same value on two independently designed radios is a **family trait, not a unit
signature**: it points at a shared design element, most plausibly the same class of integrated
transceiver used across inexpensive handhelds, rather than at an individual transmitter.

So the strongest spectral feature found here **fails as SEI on the only pair available to test
it with**. That is a negative result and it is the useful one: it is why
`esm446.analysis.eob` groups emitters by frequency and sub-audible tone and does not use the
spurious signature. Establishing that some emitter feature *does* discriminate would need
several units of the same model, which this project does not have.

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

## Measured: where the receiver's gain should be set, and why not higher

Sweeping the baseband gain with the input open, at a fixed 32 dB of front-end gain, separates
two regimes cleanly. Below about VGA 32 the noise floor barely moves with gain: the converter
dominates and the front end is being wasted. From VGA 40 upward the floor rises one decibel
per decibel, which is the condition where the input noise dominates and the receiver is doing
what it was designed to do.

| VGA | noise floor (dBFS) | rise for the gain applied |
|---|---|---|
| 0 | −88.96 | — |
| 16 | −82.99 | +6.0 dB for 16 |
| 24 | −82.03 | +6.9 dB for 24 |
| 32 | −78.57 | +10.4 dB for 32 |
| **40** | −71.39 | +17.6 dB for 40 |
| 62 | −49.79 | +39.2 dB for 62 |

Referred back to the input, the configured operating point of **VGA 20 sits about 9 dB worse
than the asymptote** reached from VGA 40 on. On sensitivity alone the gain should be raised.

**It is not raised, and the reason is measured.** At VGA 40 the receiver stops being masked by
its own converter and starts seeing the environment, which is neither Gaussian nor stationary.
Over eighteen minutes of empty band the node produced **sixteen phantom emissions** at that
gain, about one every seventy seconds, against a false alarm rate 33 times lower at VGA 20.
An order of battle assembled from that is noise with timestamps.

So the shipped default stays at LNA 32 / VGA 20 and trades 9 dB of sensitivity for an output
that can be believed. The figure is quoted here rather than buried because the trade is a
deployment decision, not a constant: a quieter site, or a narrower capture that excludes the
band edges where most of those phantoms sat, moves the balance.

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
