# ESM-446

**A passive Electronic Support Measures node for the PMR446 band.**

ESM-446 continuously surveys 445.1–447.1 MHz with a HackRF One, detects emissions with a
constant-false-alarm-rate detector, identifies them by their sub-audible signalling, estimates
emitter range with quantified uncertainty, and publishes the result to a TAK client as
Cursor-on-Target.

It records **signal metadata, never communication content** — see
[`docs/06_legal_ethics.md`](docs/06_legal_ethics.md).

> **Status: under construction.** The repository is being rebuilt from the v0 prototype preserved in
> [`legacy/`](legacy/). This README is a placeholder and will be replaced with results, figures and a
> hardware-free quickstart as the phases land.

## Why this exists

The v0 prototype worked in the sense that it produced output. It did not work in the sense that
matters: it processed signal **6.5× slower than real time**, ran the SDR at a sample rate outside the
device's specification, never actually applied its receiver gain settings, and had a 1 % sample-rate
error that pushed the CTCSS tone off its detector bin. Every one of those is measured and documented
in [`legacy/README.md`](legacy/README.md).

The rebuild exists to make the claims true, and to make them **verifiable by someone else** — which
is why the reproducible testbench (synthetic scenarios with ground truth, plus real IQ captures) is
treated as a first-class deliverable rather than an afterthought.

## Roadmap

| Phase | Content | State |
|---|---|---|
| 0 | Secure repository baseline | done |
| 1 | DSP core: polyphase channeliser, CA-CFAR detector, CTCSS bank, calibration | in progress |
| 2 | Reproducible testbench: source abstraction, scenario simulator, CI | pending |
| 3 | Metadata sinks, Electronic Order of Battle, Monte-Carlo geolocation, CoT/TAK | pending |
| 4 | Validation against real signals: conducted calibration, two-emitter acceptance test | pending |
| 5 | Systems-engineering documentation and V&V report | pending |
| 6 | Packaging: hardware-free demo, dashboard, results | pending |

## License

MIT — see [`LICENSE`](LICENSE).
