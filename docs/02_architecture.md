# Architecture — ESM-446

**Document:** ARCH-ESM446
**Version:** 1.0

How the system is put together, what flows between the parts, what it costs in time and
memory, and how it fails. Every figure here is measured on the development machine and
reproduced by `esm446-bench`; none is a design target.

---

## 1. Block diagram

```
                    ┌──────────────┐
   antenna ────────►│  IQSource    │  SoapySource (live) │ FileSource (replay) │ SimSource
                    └──────┬───────┘  one interface, and the node cannot tell which
                           │ complex64 @ 2 MS/s
                           ▼
                    ┌──────────────┐
                    │ Polyphase    │  160 channels × 12.5 kHz, 2× oversampled
                    │ filter bank  │  1920-tap prototype, designed once
                    └──────┬───────┘
                           │ (frames, 160) complex64 @ 25 kHz per channel
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌──────────────┐          ┌──────────────┐
      │  CFAR        │          │  Survey      │  STFT, low duty cycle
      │  detector    │          │  (occupancy) │  0.009 CPU-s/s
      └──────┬───────┘          └──────────────┘
             │ boolean mask
             ▼
      ┌──────────────┐
      │  Tracker     │  opens/closes emissions, 250 ms hangover
      └──────┬───────┘
             │ Emission (with retained IQ)
             ▼
      ┌──────────────┐   ┌──────────────┐
      │ NFM          │──►│ CTCSS        │  38-tone Goertzel
      │ discriminator│   │ identifier   │  audio dies here — REQ-LEG-002
      └──────┬───────┘   └──────┬───────┘
             │ deviation        │ tone, classification
             └────────┬─────────┘
                      ▼
              ┌──────────────┐
              │EmissionReport│  the node's only output
              └──────┬───────┘
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌─────────┐  ┌─────────┐  ┌──────────┐
   │ stdout  │  │ Sinks   │  │ CotSink  │──► UDP │ TCP │ TLS │ TLS server
   │ (JSONL) │  │ jsonl/db│  └──────────┘
   └─────────┘  └────┬────┘
                     │
                     ▼
              ┌──────────────┐
              │ Analysis     │  artefacts → attribution
              │ (offline)    │  eob → emitters, occupancy, bursts
              └──────────────┘  geolocation → range percentiles
```

---

## 2. Why the boundaries are where they are

**The source abstraction is the load-bearing one.** Live capture and replay differ only in
which `IQSource` is constructed. Without that, none of the test suite could exist, there
would be no demonstration for anyone without a HackRF, and CI could not run — so the
abstraction is not decoration, it is what makes the other 400 tests possible.

**Detection and analysis are separated by the store, not by a function call.** The node
decides what is an emission; everything about *which emitter*, *how many*, and *how far* is
computed later, from records. That is deliberate: attribution and grouping are judgements
about a whole set of detections, and a judgement made one detection at a time would be worse
and irreversible. It also means the archive can be re-analysed when the analysis improves,
which has already happened twice.

**Audio has no path out.** The discriminator's output is a local in `EsmNode._identify` and
is passed only to the tone detector. This is REQ-LEG-001 realised as a structural property
rather than as a configuration flag somebody could set wrongly.

**Two spectral paths, on purpose.** The filter bank is optimised for selectivity, which costs
a 1920-tap prototype. The survey is a plain STFT for occupancy and off-plan emissions at
0.009 CPU-s/s. Forcing one transform to serve both would mean paying filter-bank prices for
the wideband picture on every frame.

---

## 3. Data flow and rates

| Stage | In | Out | Rate |
|---|---|---|---|
| Source | — | complex64 | 2 MS/s, 16 MB/s |
| Channeliser | 2 MS/s | (frames, 160) complex64 | 25 kHz/channel, 32 MB/s aggregate |
| Detector | power (frames, 160) | boolean mask | same frame rate |
| Tracker | mask + IQ | `Emission` on close | event-driven, seconds apart |
| Identification | ~4 s of channel IQ | tone + deviation | once per emission |
| Report | — | one JSON object | one per emission |

The aggregate rate *rises* through the channeliser because 2× oversampling produces two
output samples per channel per input sample per channel. That is the cost of leaving room for
a real transition band, and it is paid in bandwidth rather than in CPU.

---

## 4. Timing budget

Measured, at 2 MS/s over 160 channels, in CPU-seconds per second of signal:

| Stage | Cost | Share |
|---|---|---|
| Polyphase filter bank | 0.177 | 84 % |
| CFAR detection, tracking, identification | 0.033 | 16 % |
| **Full node pipeline** | **0.210** | **4.8× real time** |
| Wideband survey (separate path) | 0.009 | — |

Two decisions dominate this budget and both were made against measurement:

**The noise estimate is held for 64 frames.** Estimating per frame builds a
`(frames, bins, references)` array — 101 MB for a single 131 ms block — and takes an order
statistic across all of it. Measured at 4.93 CPU-s/s, which made the node 4.8× *slower* than
real time with the filter bank innocent. A frame is 40 µs and a thermal noise floor is
stationary over milliseconds, so holding it is physics rather than compromise.
`test_holding_the_noise_estimate_does_not_change_the_false_alarm_rate` measures that it costs
nothing in P<sub>fa</sub>.

**The polyphase fold is chunked at 1024 frames.** The accumulator for a whole block is ~16 MB
and streams out of cache on each of the K passes. Chunking took 0.198 → 0.161 CPU-s/s for the
filter bank alone, with identical arithmetic.

CI gates on `esm446-bench --max-ratio 0.5`, so the budget is enforced rather than described.

---

## 5. Memory

Nothing scales with capture length. The three places it could have:

| Risk | Mitigation | Verified by |
|---|---|---|
| Survey accumulating every frame | Chunked accumulation, 4096 frames | `test_chunked_average_matches_a_single_pass` |
| Waterfall allocating the full spectrogram | Decimated to a cap of 8192 frames | `test_waterfall_is_capped_rather_than_allocating_gigabytes` |
| Tracker holding IQ for an emission that never ends | Hangover closes it; `filter_short` discards runts | `test_a_stuck_carrier_cannot_grow_without_bound` |

The first two are not hypothetical: an earlier revision of the survey allocated 7.21 GB on a
machine with 7.6 GB and took it down twice.

---

## 6. Failure modes

| Failure | Behaviour | Rationale |
|---|---|---|
| SDR absent or unopenable | Startup error naming the cause, exit 1 | Better than a traceback, and the replay path is unaffected |
| Misconfigured receiver geometry | **Rejected at import time**, before the SDR is opened | A mistuned band plan produces a running system quietly looking at the wrong frequencies |
| Sink fails mid-capture | Logged, skipped, capture continues | Losing the archive is bad; losing the capture is worse |
| CoT link down | Logged, message dropped, backoff 1→30 s | Nothing is queued: a buffer that grows while a link is down eventually takes the process with it |
| No calibration loaded | Warned at startup; every `estimated_dbm` is `null` | An uncalibrated estimate that looks like a measurement is worse than no estimate |
| Emission never ends | Hangover closes it after 250 ms of silence | Otherwise a carrier or a spur becomes an emission of unlimited duration |
| Local-oscillator spur | Bin excluded, before the pair test not after | A spur 31 dB over the floor drags its neighbour over too |
| Process killed mid-write | JSONL loses at most the last line | The reason for one record per line |

The pattern: **a failure of an output must never cost an input.** Capture is the only thing
that cannot be redone, so every other subsystem is allowed to fail quietly and be logged.

---

## 7. Deployment

One process, one receiver, one band. No inter-process coupling — v0's file-in-`/tmp`
handshake between a shell orchestrator and four Python scripts had a genuine race, and the
answer was to stop having two processes rather than to lock the files.

Configuration is environment-driven through `pydantic-settings` and validated at import.
Nothing else in the codebase calls `os.getenv`.

---

## 8. What is not in this architecture, and why

**No direction finding.** Needs a second coherent receiver. The consequence is documented
everywhere it matters rather than being left implicit: range without bearing, an annulus and
not a point, `ce` and not a pin.

**No database beyond SQLite.** The queries are occupancy over time and grouping by channel.
Anything larger would be infrastructure in search of a problem.

**No message queue between capture and analysis.** The store is the queue, and it is already
durable, inspectable and re-readable.
