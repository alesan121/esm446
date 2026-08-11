---
name: Bug Report
about: Report incorrect behaviour in the node or the signal chain
title: "[BUG]"
labels: bug
assignees: alesan121
---

## Description

A detailed description of what the bug is.

## Workarounds

In case there is some way that lets users avoid the issue.

## Steps to Reproduce

1. Run '...'
2. With the configuration '....'
3. Observe '....'

If the bug involves detection, identification or reported power, attach or link the IQ
capture that triggers it, or the scenario parameters that generate it. A signal processing
bug that cannot be replayed cannot be fixed.

## Expected Behavior

A clear description of what you expected to happen. Where the bug concerns a measured
quantity — power, deviation, rejection, false alarm rate — state the number you expected and
the number you got.

## Receiver Configuration

The same code behaves entirely differently across these, so a report without them is rarely
actionable.

- Source: [live SDR / file replay / simulated]
- Sample format, if replaying: [cf32, cs16]
- Sample rate: [output of `ESM446_SDR_SAMPLE_RATE_HZ`, default 2000000]
- Centre frequency: [output of `ESM446_SDR_CENTRE_FREQ_HZ`, default 446093750]
- Channeliser geometry: [`CHANNELIZER_NUM_CHANNELS` and `CHANNELIZER_DECIMATION`]
- CFAR: [`CFAR_METHOD` and `CFAR_PFA`]
- Receiver gains: [LNA, VGA, AMP]
- External gain ahead of the SDR: [external LNA, cable loss]
- Antenna: [model, and whether it covers 446 MHz]
- Calibration loaded: [yes/no; if yes, attach `calibration.yaml`]

## Environment Specifications

- OS: [Ubuntu 22.04, Windows 11, macOS 15...]
- Python version: [output of `python --version`]
- Poetry version: [output of `poetry --version`]
- numpy and scipy versions: [output of `poetry show numpy scipy`]
- SoapySDR: [version, or "not installed" if replaying from file]
- Benchmark: [output of `poetry run esm446-bench`, if the report concerns throughput]
- Logs: [output with `ESM446_LOG_LEVEL=DEBUG`; attach as a file if long]
