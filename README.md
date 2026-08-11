# ESM-446

**A passive Electronic Support Measures node for the PMR446 band.**

ESM-446 surveys 445.1–447.1 MHz with a HackRF One, splits it into 160 channels with a
polyphase filter bank, detects emissions with a constant-false-alarm-rate detector,
identifies them by their sub-audible CTCSS signalling, and reports each one as a metadata
record.

It records **signal metadata, never communication content** — see
[`docs/06_legal_ethics.md`](docs/06_legal_ethics.md). That is not a disclaimer bolted on
afterwards; it is why demodulated audio never leaves the function that identifies the tone.

## Quickstart

No SDR required. The pipeline is identical for live capture and for replay, so everything
below runs on a clone with nothing plugged in.

```bash
poetry install --with dev
poetry run pytest                    # 171 tests
poetry run esm446-bench              # throughput against the v0 baseline
poetry run esm446-node --file capture.cf32
```

Each detected emission is one JSON object on stdout:

```json
{"timestamp": 1786463892.27, "frequency_hz": 446106250.0, "pmr_channel": 9,
 "duration_s": 1.0007, "peak_power_dbfs": -25.36, "snr_db": 62.63,
 "estimated_dbm": null, "calibrated": false, "ctcss_tone_hz": null,
 "classification": "UNKNOWN", "peak_deviation_hz": 749.77,
 "gains": {"lna_db": 32.0, "vga_db": 20.0, "amp_enabled": false}}
```

Note `estimated_dbm: null`. Absolute power is reported only where a calibration exists for
that exact receiver gain configuration and level range. An uncalibrated estimate that looks
like a measurement is worse than no estimate.

## Why it was rebuilt

The v0 prototype, preserved verbatim in [`legacy/`](legacy/), produced output. It did not
work in the sense that matters, and the defects are measured rather than asserted:

| Defect | Measured | Now |
|---|---|---|
| Channeliser slower than real time | **8.66** CPU-s per signal second | **0.19** |
| Sample rate below the HackRF minimum | 800 kS/s requested, 2 MS/s minimum | rejected at startup |
| Receiver gains never applied | `8` passed as the channel index | quantised and read back |
| Audio sample-rate mismatch | 12121.2 Hz produced, 12000 assumed | derived, not restated |
| CTCSS detector loss from the above | **>12 dB** | exact-frequency Goertzel |
| Detection thresholds | two constants tuned on one afternoon | CFAR at a stated P<sub>fa</sub> |

Assembling the pipeline exposed four more that unit tests on the stages in isolation had
not, all fixed in [#9](https://github.com/alesan121/esm446/pull/9): a prototype cutoff that
passed half the adjacent channel (49.5 → 92.9 dB rejection), a CFAR noise estimate that made
the node 4.8× slower than real time, deviation readings dominated by keying transients, and
a benchmark that gated one stage while the pipeline regressed.

## Performance

Measured by `esm446-bench` on the development machine, at 2 MS/s over 160 channels:

| | CPU-s per signal second | Margin |
|---|---|---|
| v0 per-channel mixer and filter, 800 kS/s, 57 channels | 8.66 | **drops signal** |
| Polyphase filter bank | 0.19 | 5.1× real time |
| Full node pipeline | 0.26 | 3.9× real time |

The channeliser is **44× faster than v0 normalised by signal duration**, while covering 2.5×
the bandwidth and 2.8× the channels.

## Design notes

**Why the receiver tunes to 446.09375 MHz.** That is PMR446 channel 8, an exact integer
number of 12.5 kHz steps from channel 1, so every channel lands precisely on a filter bank
bin with no half-bin offset and no resampling. The midpoint of the allocation, 446.1 MHz,
looks more reasonable and is half a bin off. The configuration refuses to start on a
misaligned centre frequency.

**Why OS-CFAR rather than CA-CFAR.** PMR446 channels are adjacent by construction. A strong
emitter inside the reference window inflates a cell-averaged noise estimate and masks its
weaker neighbour; an order statistic below the interferer's rank ignores it.

**Why two spectral paths.** The filter bank is optimised for selectivity, which costs a
1920-tap prototype. The wideband survey is a plain STFT at low duty cycle for occupancy and
off-plan emissions, at 0.009 CPU-s/s. Forcing one transform to do both means over-paying for
the wideband picture on every frame.

**What the CTCSS stage is not.** It is cooperative identification by a pre-shared sub-audible
key. There is no challenge, no response and no cryptography, so it is not IFF, and calling it
that would invite exactly the wrong question.

## Roadmap

| Phase | Content | State |
|---|---|---|
| 0 | Secure repository baseline | merged |
| 1 | DSP core, detection, identification, in-process pipeline | merged |
| 2 | Scenario simulator and recorded IQ test vectors | next |
| 3 | Metadata sinks, Electronic Order of Battle, Monte-Carlo geolocation, CoT/TAK | planned |
| 4 | Conducted calibration and two-emitter acceptance test | planned |
| 5 | Systems-engineering documentation and V&V report | planned |
| 6 | Packaging: hardware-free demo and results | planned |

## License

MIT — see [`LICENSE`](LICENSE).
