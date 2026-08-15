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

## Attempted: the receiver's frequency error, and why it is a bound rather than a figure

Every frequency this system reports inherits the error of the HackRF's crystal. Consistency
between captures says the oscillator is stable and says nothing about whether it is right.

An LTE downlink does not transmit its DC subcarrier, leaving a notch about 15 kHz wide at
exactly the licensed carrier centre, which sits on a 100 kHz raster. Base stations hold
frequency to 0.05 ppm. That should make a free, accurate reference, and it does not, for
reasons that took a measurement campaign to establish.

**Result: the error is negative and smaller than 1 ppm in magnitude, which is under 450 Hz at
446 MHz. No tighter figure is claimed.**

### The estimator had a 14 % scale error, found by closed-loop injection

The first implementation took the centroid of the power deficit over a window fixed on the
nominal frequency. Shifting a real capture by a known amount and re-measuring it recovered
**86 % of every shift**, on every capture tried:

| injected | recovered | gain |
|---|---|---|
| −2000 Hz | −1700 Hz | 0.850 |
| −500 Hz | −420 Hz | 0.839 |
| +500 Hz | +433 Hz | 0.866 |
| +2000 Hz | +1726 Hz | 0.863 |

A centroid taken over a window fixed on the nominal frequency contracts towards the window
centre as the notch moves away from it. The unit tests of the day did not catch it because
their tolerance, 150 Hz on shifts of a few hundred hertz, was wider than the error itself.

The replacement estimates the notch's **axis of symmetry** — the offset at which the profile
best matches its own mirror image. That has unity gain by construction, and measured the same
way its worst error is 8.7 Hz over ±3 kHz, a gain error of 0.29 %.

### One capture was contaminated by the receiver's own local oscillator

A capture tuned onto the carrier read **−14 Hz** where six other local oscillators on the same
carrier read −109 to −367 Hz. The local-oscillator spur sits at baseband zero, lands inside
the notch, and pulls the estimate towards it. That biases the receiver towards looking better
calibrated than it is, so it would never announce itself as an outlier. The tool now refuses
any capture tuned within 200 kHz of the carrier.

### Four references, and they do not agree

Six local oscillators per carrier, four carriers spanning a factor of 2.3 in frequency. A
crystal error is a constant in parts per million, so all four must agree:

| carrier | band | measured |
|---|---|---|
| 806.0 MHz | 20 | −0.351 ± 0.009 ppm |
| 816.0 MHz | 20 | −0.624 ± 0.049 ppm |
| 940.0 MHz | 8 | −0.168 ± 0.033 ppm |
| 1835.1 MHz | 3 | −0.410 ± 0.008 ppm |

χ²/dof = **30**. They disagree by a factor of 3.7, an order of magnitude beyond their own
scatter. The transmitters cannot explain it: 0.05 ppm at 816 MHz is 0.04 Hz.

### What is actually limiting it

Repeating the measurement on one carrier at a **fixed** local oscillator, ten captures over
two minutes, separates the receiver from the reference:

| carrier | scatter over 10 repeats |
|---|---|
| 806.0 MHz | 24 Hz (0.030 ppm) |
| 816.0 MHz | 64 Hz (0.078 ppm) |
| 940.0 MHz | 130 Hz (0.138 ppm) |
| 1835.1 MHz | 93 Hz (0.051 ppm) |

Nothing about the receiver changed between those captures. What changed is the traffic on the
subcarriers either side of the notch, which is what the symmetry estimate is comparing. The
notch is a gap in a live, loaded signal, not a marker on a quiet one.

Screening the references on that basis — a reference is usable only if its own repeats agree
to better than 0.05 ppm, a criterion about the reference and not about whether it agrees with
the others — **rejects three of the four**. Only 806.0 MHz passes, at −0.343 ± 0.009 ppm, and
it is repeatable to 11 Hz across three separate sessions. One surviving reference with nothing
independent to check it against is not a calibration, so it is recorded as indicative and not
quoted as the receiver's error.

### A second site supports the traffic explanation, and does not settle it

A shorter run (four local oscillators, not six) from a second site with 18–25 dB more SNR on
both carriers than the first campaign:

| carrier | measured | scatter |
|---|---|---|
| 816.0 MHz | −0.011 ± 0.007 ppm | tight |
| 940.0 MHz | −0.137 ± 0.051 ppm | wide |

If the problem were the local oscillator or the receiver, more signal would not have changed
anything. It did: 816.0 MHz's own scatter fell by roughly an order of magnitude at the
stronger site, while 940.0 MHz stayed noisy — which is what "sensitive to the neighbouring
subcarrier traffic, not to SNR" predicts, since traffic loading is independent of link budget.
The two carriers still disagree, 0.126 ppm against a combined statistical uncertainty of
0.051. This is consistent with everything above and does not replace it: four points per
carrier is half the first campaign's depth, and the honest result stays what it was --
**negative, under 1 ppm, bounded and not calibrated.**

A GSM FCCH burst was tried as an independent method with entirely different systematics and
returned −1.65 ppm, disagreeing with all four notch measurements. That disagreement is
unresolved and is itself a reason not to quote a figure.

### The DVB-T method worked, once, and does not settle it either

The band-edge method this project could never validate before -- no multiplex was ever
receivable at usable strength -- finally measured something, once the telescopic antenna was
cut for the DVB-T band (12.9 cm, a quarter wave at 580 MHz) instead of 446 MHz. UHF channel 28
(530 MHz) gave **−0.14 ppm**, with the method's own edge-agreement uncertainty at ±0.39 ppm --
wide, but consistent with the surviving LTE reference (806.0 MHz, −0.343 ± 0.009 ppm) and with
the overall bound.

![The DVB-T block on channel 28, flat top and sharp edges landing exactly on the shaded
occupied bandwidth](figures/08_dvbt_channel28_spectrum.png)

*Unlike the other figures in this document, this one is not reproduced by `make report`: it
is a single spectrum from the capture this measurement was made on, not something a
simulation can stand in for. Kept as a committed image for that reason, with the campaign
that produced it described here rather than a script that regenerates it from data nobody
without this exact hardware session has.*

The repeatability check this project applies to every frequency method -- retune and
re-measure -- was attempted three times before it produced a real number, and each failure is
worth recording because each pinned down what was actually wrong.

**`_edge()` starts from the spectrum's global peak and walks outward**, so the failure mode is
not "too close to a fitted window" but simpler and sharper: if the receiver's own
local-oscillator leakage lands *inside the channel's own occupied bandwidth*, it becomes the
peak the edge-search starts from, and the result is nonsense. At 10 MS/s, retuning 2 MHz put
the leakage at 528 MHz -- inside the 526.19-533.8 MHz occupied span -- and returned an
edge-agreement of ±405 kHz, correctly not mistaken for a measurement. Retuning 4 MHz moved the
leakage clear of the channel but also moved the channel's low edge outside the ±5 MHz Nyquist
window, and the width guard correctly refused a truncated block.

At 10 MS/s no local-oscillator position satisfies both constraints (leakage outside the
channel, both edges inside Nyquist) at once. **20 MS/s does**, and a capture at 525 MHz --
5 MHz off-channel, comfortably outside both the occupied span and short of the Nyquist edge --
finally gave a second real measurement: −2.41 ppm, edge-agreement ±2191 Hz.

That is not a clean confirmation of the on-channel −0.14 ppm, and it is reported as what it
is rather than rounded up to one: the two are compatible only because ±2191 Hz is a wide
uncertainty (±4.1 ppm at 530 MHz), not because they agree tightly. The reason is the sample
count, not the geometry fix. Widening the analysis to hold everything `average_spectrum`
builds in memory -- the raw array, the windowed copy, and the FFT output, each the size of the
capture -- means memory scales directly with sample count, not with the FFT size. 90 million
samples (nine seconds at 10 MS/s) is what produced the tight −0.14 ppm figure and used most of
this machine's 7.6 GB; **190 million samples at 20 MS/s drove it to 6.9 GB resident and into
swap thrashing**, and was killed rather than left to find out whether it would recover. Sixteen
million samples (0.8 s) is what actually ran, at a fraction of the averaging the first
measurement had, which is the whole difference in precision. `esm446-calibrate-frequency`
defaults to four million samples for exactly this reason; overriding it without checking
memory headroom first is the mistake here, not the tool.

The DVB-T tool still has no equivalent of the notch method's 200 kHz minimum-offset guard, and
now has a second, sharper requirement it does not check either: the local oscillator must
clear the channel's own occupied bandwidth, not just sit some fixed distance from centre. Both
are gaps in the tool, filed rather than patched here under time pressure. Two real, mutually
compatible-but-not-confirming measurements are what this project has for DVB-T; a third,
adequately averaged one is what would settle whether −0.14 ppm or −2.41 ppm is closer to the
truth, and it needs more memory than this machine safely offers at 20 MS/s.

### A properly averaged campaign found a third number, and a receiver artefact

Processing a real capture in memory-bounded pieces rather than all at once -- the lesson of
the incident above -- makes long averaging possible without the memory ceiling: 20 minutes at
525 MHz, read as independent ~240 MB segments, each measured and released before the next is
opened, peak resident memory 1.4 GB regardless of the campaign length.

Doing this surfaced a receiver artefact this project had not seen before, because nothing
before ran long enough to hit it: **a spur at exactly local-oscillator-minus-quarter-sample-rate**
(525 − 5.000 = 520.000 MHz here, a suspiciously round baseband offset for a real transmission)
that is comparably strong to the DVB-T block itself -- 24.6 dB against the channel's 24.4 dB
in one segment -- and wins `_edge()`'s peak search close to half the time. Of 36 segments
across the 20 minutes, 20 locked onto this artefact and were excluded on that basis, uniformly
and before looking at what they would have measured, not after.

The 16 that survived:

| | value |
|---|---|
| span | 19.9 minutes |
| mean offset | −3230.7 Hz |
| standard error of the mean | ±38.9 Hz |
| in ppm | **−6.10 ± 0.07 ppm** |
| occupied width, mean ± std | 7620.3 ± 0.5 kHz (expected 7610) |
| drift over the campaign | −10.8 Hz/min, not significant against the point-to-point scatter |

That is the tightest statistical uncertainty any frequency measurement in this project has
produced. **It is not reported as an improved figure, because it disagrees with the on-channel
−0.14 ppm measurement by roughly 6 ppm** -- thirty times either measurement's own stated
uncertainty. Both cannot be right. Screening out one identified artefact does not mean no
others remain in either geometry, and an on-channel capture with the receiver's own leakage
sitting exactly at the point being measured is exactly the kind of setup this project has
repeatedly found to bias results in ways that are not obvious until measured. Which of the two
numbers -- or neither -- is closer to the truth is not established here.

**The honest position is unchanged from before this campaign, and is now better evidenced
rather than settled:** the DVB-T method, like the LTE notch method, produces numbers that
depend on the measurement geometry in ways not yet fully characterised. `measure_centre()`'s
peak-then-walk design has no defence against a comparably strong feature elsewhere in the
window, which is a real gap in the tool and not something to patch under time pressure between
one finding and the next.

### What would settle it

A GPS-disciplined oscillator, either as the HackRF's clock input or as a reference transmitter.
That is the same class of instrument the power calibration needs
([#41](https://github.com/alesan121/esm446/issues/41)) and it is the honest answer: this
measurement is limited by not having a traceable reference, not by the technique.

The tooling stays in the repository because it is now verified — unity gain, tilt immunity,
noise rejection, and a refusal to measure a contaminated capture — and because the bound it
establishes is genuinely useful: whatever the crystal is doing, it is doing it at well under
a part per million, which is far better than the specification permits and small enough not to
disturb emitter grouping, where the tolerance is 3 kHz.

## Measured: a real off-grid emitter, found indoors and confirmed by retune

A two-hour capture at the shipped default configuration (446.593750 MHz, LNA 32 / VGA 20),
taken indoors -- a new environment for this project, everything before it was outdoors --
surfaced a peak just below 446.000000 MHz, six kilohertz below the first PMR446 channel
centre and therefore off-grid rather than on any nominal channel.

Retuning the local oscillator to 446.093750 MHz is the same test used throughout this
project to separate a real signal from a receiver artefact, and it gives a clean answer:

| local oscillator | peak stayed at | verdict |
|---|---|---|
| 446.593750 MHz | 445.999377 MHz | — |
| 446.093750 MHz | (present, same feature) | **unchanged: real** |

The strongest peak in each capture is not this one -- it is the receiver's own local-oscillator
leakage, and it moved with the retune exactly as documented above (446.593750 -> 446.093750
MHz), which is what confirms the method is working rather than coincidental.

The node's own detector, run over the full two hours (`esm446-node --file`, the shipped
CFAR pipeline, not a bespoke script), found the same emitter independently and reports it
with metadata the manual spectral check does not give:

| timestamp offset | frequency | duration | SNR | peak deviation |
|---|---|---|---|---|
| 44.5 min | 445.999377 MHz | 110 ms | 2.8 dB | 12.27 kHz |
| 70.2 min | 445.999394 MHz | 125 ms | 3.1 dB | 12.20 kHz |

Both reports agree on frequency to 17 Hz, 25.7 minutes apart, which is consistent with one
intermittent transmitter rather than two coincidental events. Both are marked `off-grid` and
`UNKNOWN` (no CTCSS), correctly: it is not a nominal PMR446 channel and carries no cooperative
identification tone.

**These are also the only two emissions the detector reported over the entire two hours.**
At the CFAR design point of 1e-8 per cell, over 160 channels at a 25 kHz frame rate for
7200 seconds -- 2.88e10 cells -- zero false alarms were observed: both detections attribute
to the one confirmed real emitter, not to noise. That is a much longer, cleaner run than
what REQ-FUN-003 could previously cite, and it was taken indoors, a harder environment than
the outdoor tests that found an 8e-6 crossing rate at high gain. It does not raise the
design point to a proven figure -- absence of a false alarm in one finite run bounds the
rate, it does not measure it -- but it is the strongest evidence yet that the shipped
default (LNA 32 / VGA 20) earns its 9 dB of foregone sensitivity.

What the 445.999 MHz emission is has not been established, and is not guessed at here: an
indoor residential environment has far more candidate sources than an outdoor PMR446 test, and
attributing it to any one of them without evidence would be exactly the kind of claim this
document exists to avoid making. It is reported as what was measured -- a real, off-grid,
persistent emission -- which is the survey capability `esm446.analysis.eob` and REQ-FUN-007
exist for.

## Measured: real-world throughput runs higher than the synthetic benchmark

The same two-hour run gives a second figure for free: `esm446-node` logs its own cost on
completion, over real signal rather than the synthetic noise `esm446-bench` uses. It spent
2725.3 CPU-seconds on 7200.0 signal-seconds -- about 26 % more per signal-second than the
synthetic benchmark's median (see `docs/02_architecture.md`), a single real run against a
median of five, not two values for the same measurement.

Worth taking at face value rather than explained away: two real detections each cost
tracking and identification work the synthetic benchmark's quiet band never exercises, and
the CFAR reference-cell arithmetic runs on real receiver statistics rather than the
benchmark's synthetic ones. REQ-PER-001 requires at least 2x real-time margin; this run
still clears 2.6x, comfortably inside it but with less headroom than the synthetic figure
implies. One run carries no error bar -- the project's own measured run-to-run variance is
up to 45 % -- so this is a data point in the same direction as the benchmark, not a
replacement for it.

## What remains unmeasured

- **Absolute power.** Blocked on [#41](https://github.com/alesan121/esm446/issues/41).
- **Antenna gain.** The link budget uses a conservative default rather than a supplier figure,
  because those figures are not physical at this frequency — see `esm446/core/antenna.py`.
  Gain by substitution against a quarter-wave reference needs no extra equipment and has not
  been done.
- **Frequency accuracy, to better than a bound.** Established as under 1 ppm and negative;
  see the section above for why four cellular references disagree by more than that bound is
  tight. Needs a disciplined oscillator.
- **Achieved image rejection.** Where the images fall is computed; how far down they are is
  not measured.
