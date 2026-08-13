# Verification and validation report — ESM-446

**Document:** VV-ESM446
**Version:** 1.0
**Configuration under test:** 2 MS/s, 160 channels of 12.5 kHz, OS-CFAR at P<sub>fa</sub> 1e-8

Every requirement in [`01_requirements.md`](01_requirements.md), what verifies it, and what
the verification measured. Every figure below is regenerated from the system itself by

```bash
poetry run esm446-vv
```

which runs the real channeliser, detector and estimator and writes the underlying numbers to
`figures/results.json`. Nothing here is drawn by hand, and the report's tables and its plots
are read from the same file so they cannot disagree.

**Summary:** 45 requirements. **42 MET**, **2 PARTIAL**, **1 BLOCKED**. The blocker is a
calibrated power source; every consequence of not having one is traced below rather than
absorbed.

---

## 1. Verification methods

| Method | Meaning | Used for |
|---|---|---|
| **T** — Test | Executed against the implementation, pass/fail | 44 requirements — every one not blocked |
| **A** — Analysis | Derived, and cross-checked against an independent implementation in `matlab/` | link budget, channel plan, threshold factors |
| **I** — Inspection | Established by reading the code, where the property is structural | audio containment, bind address |
| **D** — Demonstration | Shown end to end on a clean checkout | replay pipeline |

`tests/test_requirements.py` enforces the matrix: a requirement with no verifying test, or
naming one that does not exist, fails the build. The traceability below is therefore a build
artefact rather than a document somebody maintains by hand.

---

## 2. Channelisation

![Prototype filter response](figures/01_channel_response.png)

| Requirement | Result |
|---|---|
| REQ-FUN-001 — channels on the 12.5 kHz raster | **MET.** Bin mapping agrees with the independent Octave derivation in `matlab/channel_plan.m` |
| REQ-FUN-002 — ≥60 dB adjacent-channel rejection | **MET — 92.9 dB measured on a modulated emitter; 94.1 dB from the prototype's response alone** |

The rejection figure has history worth keeping. An earlier revision placed the prototype's
−6 dB point midway between the channel edge and the alias limit, reasoning that this "used"
the available transition band. It does — by passing the inner half of the adjacent channel.
Measured on a modulated emitter, rejection was **49.5 dB**. Moving the cutoff back to the
channel edge took it to **92.9 dB** at no CPU cost, and a strong emitter stopped producing
spurious detections in both neighbouring bins. A channeliser fault that looks exactly like a
detector fault.

---

## 3. Sensitivity against offset from a bin centre

![Sensitivity ripple](figures/02_sensitivity_ripple.png)

| Requirement | Result |
|---|---|
| REQ-FUN-004 — state the penalty for an off-centre emitter | **MET — measured −6.02 dB worst case** |

Flat to within a hundredth of a decibel across 60 % of the bin, then a cliff. This is not the
gentle sag of a windowed FFT's scalloping loss; it is the prototype's transition band, and
the worst case is the same −6 dB point that buys the 94 dB rejection above. **The two numbers
are the same design decision seen from either side.**

The summed-pair curve shows why the obvious fix returns less than it appears to: the split
energy does come back, 6.02 dB becoming 3.01 dB, but the noise in the second bin comes with
it. The detection gain is 2.07 dB from reduced variance, not 3 dB from recovered energy, and
enabling it made the recorded captures worse — see [`../esm446/core/detector.py`](../esm446/core/detector.py)
and the closing note in §9.

**So the honest sensitivity figure for this system carries a 6 dB ripple.** Quoting the
on-centre value alone would overstate it by that much, for exactly the emitters it is least
likely to already know about.

---

## 4. Detection

![False alarm rate](figures/03_false_alarm_rate.png)

| Requirement | Result |
|---|---|
| REQ-FUN-003 — P<sub>fa</sub> independent of noise level | **MET.** Design 1e-3; measured 0.90e-3 to 1.12e-3 across noise levels spanning **nine orders of magnitude** |
| REQ-PER-005 — holding the noise estimate does not move the rate | **MET.** Verified at update intervals 1, 64 and 256 |

This is the property v0's fixed thresholds could not have had. `MIN_POWER = 0.065` encoded one
antenna, one gain setting and one afternoon's noise floor; moved anywhere else it either went
deaf or fired constantly, and neither failure announces itself.

![Probability of detection](figures/04_detection_probability.png)

| | SNR for P<sub>d</sub> = 0.5 |
|---|---|
| On a bin centre | **15.0 dB** in-channel |
| Half a bin off centre | **20.0 dB** in-channel |

Both at P<sub>fa</sub> = 1e-8 per cell per frame — which at 160 bins and a 25 kHz frame rate
is four million cells a second, so 1e-8 is roughly one false alarm every three hours rather
than the 400 a second a textbook 1e-4 would produce here.

---

## 5. Identification

| Requirement | Result |
|---|---|
| REQ-FUN-005 — identify any of the 38 standard CTCSS tones | **MET.** Every tone identified through the full chain |
| REQ-FUN-006 — FRIEND only on the configured tone | **MET.** Verified on two real handsets carrying different codes |
| REQ-FUN-008 — measure peak deviation | **MET.** 1347 Hz measured on a real transmission, inside the 2.5 kHz ETSI narrowband limit |

The strongest result here came from the recorded two-emitter capture: the node identified the
second handset's **141.3 Hz** tone *before* the operator confirmed what it had been set to.
That is the only version of this test that counts, since anything else checks that the answer
was known in advance.

The generalised Goertzel is what makes it work. v0 used a fixed 12000-sample block against an
audio path that actually ran at 12121.2 Hz, so 114.8 Hz landed 1.15 bins off its own detector
— **more than 12 dB of loss**, appearing as spurious "unknown" classifications.

---

## 6. Attribution and order of battle

| Requirement | Result |
|---|---|
| REQ-FUN-011 — aggregate into an order of battle | **MET** |
| REQ-FUN-012 — counts are lower bounds without overlap | **MET** |
| REQ-FUN-013 — attribute by-products to their emitter | **MET — 11 emitters → 3 on the recorded captures** |
| REQ-FUN-014 — never delete an attributed detection | **MET** |

Two handsets produced twelve detections. Un-attributed, the order of battle reported eleven
emitters. Attribution by arithmetic relation — a pair symmetric about one carrier, or a
third-order product of two at 2·f₁−f₂ — reduced that to three, of which two are the radios
and one is a 0.36 s detection at 2.9 dB SNR that nothing explains and which is therefore not
claimed to be explained.

| relation | predicted | measured | error |
|---|---|---|---|
| splatter symmetry | — | — | **+211 Hz**, **−14 Hz** |
| 2·f₁−f₂ | 445.968 813 MHz | 445.968 917 MHz | **+104 Hz** |
| 2·f₂−f₁ | 446.156 190 MHz | 446.156 089 MHz | **−101 Hz** |

Against a 12.5 kHz channel spacing.

---

## 7. Range estimation

![Interval coverage](figures/05_interval_coverage.png)

| Requirement | Result |
|---|---|
| REQ-FUN-016 — Monte Carlo uncertainty, empirical percentiles | **MET** |
| REQ-CAL-003 — refuse a range from uncalibrated power | **MET** |
| REQ-CAL-005 — intervals contain the truth at the declared rate | **PARTIAL** |

Measured over 300 realisations:

| declared | achieved |
|---|---|
| 5 % | 5.3 % |
| 50 % | 50.3 % |
| 68 % | 69.0 % |
| 90 % | 89.7 % |
| **95 %** | **95.3 %** |

**Why this is PARTIAL and not MET.** The realisations are drawn from the same prior the
estimator assumes. This verifies that the estimator inverts its own model correctly — a real
property, and one that is easy to get wrong — but it would look exactly this good if the prior
were wrong about the environment. Validating the prior needs measured distances against
measured power, which needs REQ-CAL-004, which is blocked.

### How badly it fails when the prior is wrong

![Coverage under misspecification](figures/07_misspecification.png)

Rather than leave that as a caveat, the exposure is measured: hold the estimator's assumptions
fixed and move the *true* path loss exponent away from the assumed 3.5.

| true exponent | 50 % ring | 95 % ring |
|---|---|---|
| 2.50 — near free space | 0 % | **34 %** |
| 2.75 | 0.5 % | 74 % |
| 3.00 | 3 % | 94 % |
| 3.50 — as assumed | 42 % | 100 % |
| 4.00 | 95 % | 100 % |
| 4.50 | 100 % | 100 % |

**The failure is asymmetric, and that is the useful part.** If the environment is more
obstructed than assumed, the estimator places the emitter further out than it is and the rings
still contain it — over-covering costs precision, not correctness. If the environment is
*clearer* than assumed, the emitter is further away than the model can reach and the ring
simply does not extend to it. In an open field, an exponent nearer 2.5, a ring claiming to
hold the emitter 95 % of the time holds it **about a third** of the time.

The operational consequence is a single sentence: **assume more obstruction than you think you
have, never less.** Three tests pin it, including one asserting that the conservative error is
the one in that direction, so the conclusion cannot be softened by editing this paragraph.

**What the estimate is worth.** For a signal at −95 dBm the median is 572 m and the 5–95 %
interval spans a factor of **42**. Taking each uncertainty alone:

| assumption | 5–95 % span it produces by itself |
|---|---|
| **path loss exponent, ±0.5** | **×25.0** |
| shadowing, 8 dB | ×5.6 |
| emitter EIRP, ±3 dB | ×1.9 |
| receiver calibration, ±2 dB | ×1.5 |

The exponent decides the answer and everything else is detail. Two consequences follow, and
both are the kind of thing a V&V report exists to surface: a range from one omnidirectional
sensor is nearly worthless without a locally measured exponent, and **calibrating the
receiver would improve it by almost nothing** — which is the opposite of what the v0
estimator's single shadowing sigma implied.

---

## 8. Interfaces and performance

![Throughput](figures/06_throughput.png)

| Requirement | Result |
|---|---|
| REQ-PER-001 — real time with ≥2× margin | **MET — 0.23 CPU-s/s median, 4.4× real time** |
| REQ-PER-002 — channeliser under 0.5 CPU-s/s | **MET — 0.17 median** |
| REQ-PER-003 — survey under 5 % of the channeliser | **MET — 0.009** |
| REQ-PER-004 — memory does not scale with capture length | **MET** |
| REQ-IF-001 — CoT validates against the schema | **MET** |
| REQ-IF-002 — the message does not depend on the transport | **MET.** Byte-identical over UDP, TCP and TLS |
| REQ-IF-004 — a failed publication does not interrupt capture | **MET** |
| REQ-IF-005/006 — range without bearing, unknown when uncalibrated | **MET** |
| REQ-IF-008/009 — records round-trip; older stores stay readable | **MET** |

v0 needed **6.9** CPU-seconds per signal second at a quarter of the bandwidth and a third of
the channels, so roughly **85 % of the incoming signal was never examined**.

Every throughput figure here is the **median of five runs**, and the range is published in
`figures/results.json` alongside it. One run of this pipeline varies by up to 45 % with
machine load — the full node ranged 0.23 to 0.34 across the five taken for this report — so a
single measurement quoted as *the* number is false precision. CI gates on the measurement
rather than on this table, which is what stops the two drifting apart.

---

## 9. What is not verified, and why

Three shortfalls, stated in full because a V&V report that lists only successes is an
advertisement.

**REQ-CAL-004 — absolute power. BLOCKED.** No calibrated source is available, so every
`estimated_dbm` the node produces is `null` and every range estimate is marked uncalibrated.
The consequences are traced rather than hidden: `ce` in the CoT message carries the unknown
sentinel, `estimate_from_report` returns `None`, and the link budget's predicted sensitivity
figures are labelled as computed from datasheet values for a device that has not been
characterised. See [#41](https://github.com/alesan121/esm446/issues/41).

**REQ-CAL-006 — frequency accuracy. PARTIAL.** Reported frequencies are consistent to within
tens of hertz across captures, but nothing establishes that they are *correct*. The HackRF's
crystal drifts by several parts per million, which at 446 MHz is hundreds of hertz of
systematic error. Calibrating against a broadcast carrier would settle it and has not been
done.

**Specific emitter identification — attempted, and it failed.** The ±37.5 kHz spurious pair
was the strongest candidate: discrete, repeatable, and a property of the transmitter rather
than of the path. Measured across a Baofeng UV-5RA and a Radtel RT-900 it sits between −33.4
and −34.7 dBc — a spread inside the measurement's own uncertainty. A feature taking the same
value on two independently designed radios identifies a family, not a unit. No requirement
was levied on it, and the order of battle groups on frequency and tone instead.

One further negative result worth recording: the **pair-detection test** described in §3 was
implemented, verified, measured to recover 1.25 dB at the worst offset for 0.38 dB everywhere
else — and then measured on the recorded captures to push a genuine sideband detection below
the threshold, breaking the attribution in §6. It ships **off**. A synthetic tone said modest
win; the real captures said no, and the real captures decided it.

---

## 10. Traceability summary

| Category | Requirements | MET | PARTIAL | BLOCKED |
|---|---|---|---|---|
| Functional (REQ-FUN) | 17 | 17 | — | — |
| Performance (REQ-PER) | 5 | 5 | — | — |
| Interface (REQ-IF) | 9 | 9 | — | — |
| Calibration (REQ-CAL) | 6 | 3 | 2 | 1 |
| Legal and ethical (REQ-LEG) | 5 | 5 | — | — |
| Configuration (REQ-CFG) | 3 | 3 | — | — |
| **Total** | **45** | **42** | **2** | **1** |

No requirement is orphaned: every one names a verifying test or states its blocker, and
`tests/test_requirements.py` fails the build if that stops being true.
