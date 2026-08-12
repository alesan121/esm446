# RF analysis scripts

The engineering calculations behind the design decisions, kept so the working survives rather
than living in a commit message.

```bash
cd matlab
octave-cli --no-init-file esm446_analysis.m
```

Written in MATLAB syntax and verified under **GNU Octave 6.4**. `--no-init-file` is worth
using: a personal `.octaverc` that changes directory or extends the path will otherwise
shadow these files.

## What is here

| Script | Question it answers |
|---|---|
| `link_budget.m` | What does the receive chain hear, and what does the external LNA buy? |
| `measurement_setup.m` | How close can a transmitter be before the receiver stops being linear? |
| `channel_plan.m` | Where do the channels, the DC spur and the IQ images land? |
| `esm446_analysis.m` | Runs all three and prints the report. |

## Why these are separate from the Python

The Python under `esm446/core/` is the authority: it is what the node executes, and its
figures are the ones the system acts on. These scripts are **analysis**, which is a different
job — working out what the design should be, and recording why.

Keeping them as an independent implementation of the same physics has a practical benefit
beyond documentation. `tests/test_matlab_consistency.py` runs both and compares them, so a
disagreement is a test failure rather than a coincidence nobody notices. Two people deriving
the same number two ways is the cheapest review available.

## Results these produce

Noise figure and sensitivity, at 446 MHz in a 12.5 kHz channel:

| | without LNA | with 20 dB LNA |
|---|---|---|
| cascaded noise figure | 8.50 dB | **1.69 dB** |
| noise floor | -124.5 dBm | -131.3 dBm |
| minimum detectable signal | -111.5 dBm | -118.3 dBm |

Friis is the whole story: the first stage sets the noise figure, so 20 dB of gain at the
antenna divides the receiver's own 8 dB contribution by a hundred. That is 6.8 dB, which
under a log-distance exponent of 3.5 is 57 % more detection range.

Near-field test setup, handset at minimum power, 3 m indoors:

| | received | verdict |
|---|---|---|
| antennas aligned | -5.0 dBm | exactly at the linear limit, compressing |
| antennas cross-polarised | **-25.0 dBm** | 20 dB of margin, usable |

Distance is the obvious lever and indoors it is the one you do not have. Turning the antennas
at right angles is worth more than tripling the separation, and costs nothing.

Channel plan, comparing the centre frequency that was chosen first with the one that shipped:

| centre | grid aligned | DC spur | IQ images onto channels |
|---|---|---|---|
| 446.093750 MHz (channel 8) | yes | **on PMR8** | **15 of 16** |
| 446.593750 MHz (offset-tuned) | yes | outside the band | **0** |

Both satisfy grid alignment, which was the only constraint considered at first. Only one
keeps the receiver's own artefacts out of the allocation.

## Assumptions

Every figure here rests on values that have not been measured yet, and they are marked as
defaults in each script rather than hidden:

- LNA gain and noise figure are datasheet-typical, not measured.
- Handset EIRP is assumed at 30 dBm — 1 W at minimum power into a rubber antenna of roughly
  0 dBi.
- Cross-polarisation isolation is taken as 20 dB, which is conservative for real antennas.
- The HackRF's linear limit is taken as -5 dBm and its damage threshold as +10 dBm.

Free-space loss is also only meaningful in the far field. Below three wavelengths — 2.0 m at
this frequency — the scripts floor their answers and say so, because the formula stops
describing anything real down there.
