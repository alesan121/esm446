# Legacy code (v0 baseline)

This directory preserves the original **"SIGINT Node v16.4"** implementation exactly as it stood
before the rewrite, minus redacted credentials. It is kept deliberately: the delta between this code
and the current system is the engineering story of the project, and the V&V report cites the
before/after channeliser benchmark measured against `channelizer.py` in this directory.

**Nothing here is on the active execution path.** Do not run it.

| File | Role in v0 | Superseded by |
|---|---|---|
| `channelizer.py` | 57 independent mixer/FIR channels, 800 kHz | `esm446/core/channelizer.py` (polyphase filter bank) |
| `EW_FoF_Scanner.sh` | Bash orchestrator, file coupling via `/tmp` | `esm446/cli/node.py` |
| `range_estimator.py` | Log-distance range + CoT rings | `esm446/core/geolocation.py` |
| `iff_detector.py` | Single-tone 114.8 Hz Goertzel | `esm446/core/ctcss.py` (38-tone bank) |
| `cot_server.py` | Standalone TCP/TLS CoT server | `esm446/io/cot_transport.py` |
| `nbfm_rx.py` | GNU Radio NFM receiver, never in the pipeline | — (reference only) |
| `squelch.py`, `rms_meter.py`, `iff_test.py` | Development scratch tools | — |

## Known defects in v0

These are documented because the rewrite exists to fix them, and because a defect you can quantify is
worth more than one you merely assert. Each is reproduced by a regression test in `tests/`.

1. **Not real-time, by 6.5×.** With v0 parameters (57 channels, 262144-sample block at 800 kHz =
   0.328 s of signal), one block took **2.14 s** to process on the development machine. The
   channeliser rebuilt a 101-tap FIR via `firwin` on every call and ran a full-length `lfilter` per
   channel. Roughly 85 % of the incoming signal was dropped.
2. **Sample rate outside HackRF specification.** `SAMP_RATE = 800000` is below the device minimum
   (the project's own `scan.log` enumerates 1 MHz as the lowest offered rate; libhackrf operates
   2–20 MHz). The driver did not deliver what the code assumed, so the entire channel grid was
   mistuned.
3. **Gains never applied.** `sdr.setGain(SOAPY_SDR_RX, 8, "LNA", GAIN_LNA)` passes `8` as the
   *channel index*, not a gain value. The HackRF exposes only channel 0.
4. **Audio sample-rate mismatch.** `int(800000/12000) = 66`, so the true output rate was 12121.2 Hz
   while the CTCSS detector and the encoder both assumed 12000 Hz. The 114.8 Hz tone landed at
   113.65 Hz, roughly 1.15 bins off the Goertzel target, degrading identification sensitivity.
5. **Race condition in the orchestrator.** The dispatch loop tested only for the *existence* of
   `raw_*.s16`, not for the write having completed.
6. **Hardcoded credentials.** A Telegram bot token and chat ID were literals in the shell script
   (redacted here; the token has been revoked).
