# ICD-ESM446-CoT — Interface Control Document

**Interface:** ESM-446 sensor node → Cursor-on-Target consumer (TAK server, ATAK, iTAK, WinTAK)
**Version:** 1.0
**Status:** implemented and tested; see `tests/test_cot.py` and `tests/test_cot_transport.py`

This document specifies everything that crosses the boundary between this system and a TAK
client: the messages, every field with its unit, when messages are sent, how long they remain
valid, what happens when the link fails, and what the receiving side must not conclude from
them. It is normative — where this document and the code disagree, one of them is a defect.

---

## 1. Scope and participants

| | |
|---|---|
| **Producer** | `esm446` node, one instance per receiver |
| **Consumer** | Any CoT-speaking client. Developed against iTAK; nothing is client-specific |
| **Direction** | One-way, producer to consumer. Nothing is read back |
| **Encoding** | UTF-8 XML, one `<event>` document per message, no framing between documents |
| **Schema** | `schemas/cot-event.xsd`, validated in CI |

The producer never accepts commands, never reads from the socket, and has no concept of a
subscription. A consumer that connects to the TLS server receives whatever is published from
that moment on; nothing is replayed.

---

## 2. Transports

Selected by configuration (`ESM446_COT_DESTINATION` or `--cot`). The message is identical on
all of them — built once, then handed to the transport as bytes — which
`test_the_same_emission_is_identical_over_every_transport` asserts.

| Scheme | Form | Delivery | Use when |
|---|---|---|---|
| `udp://host:port` | datagram | none, no retry | a local TAK server or a multicast group |
| `tcp://host:port` | stream, client | ordered, acknowledged | the consumer accepts plain TCP |
| `tls://host:port` | stream, client | ordered, acknowledged, encrypted | a TAK server, which normally requires TLS |
| *(server)* | stream, server | ordered, acknowledged, encrypted | iTAK, which dials the sensor |

Default port **4242**. No destination configured means nothing is published: an unconfigured
node must not put traffic on a network.

**Certificate policy.** The TLS client verifies the server certificate by default. Disabling
verification is an explicit argument and logs a warning stating that the far end is no longer
authenticated. The TLS server does not request a client certificate, because a TAK client on a
private deployment does not present one.

---

## 3. Messages

### 3.1 Emitter track

One per completed emission.

| Attribute | Value | Unit / format |
|---|---|---|
| `version` | `2.0` | — |
| `uid` | `ESM446.<PMR<n>\|<freq_hz>>.<tone\|notone>` | stable per emitter |
| `type` | `a-f-G` or `a-u-G` | CoT type hierarchy |
| `how` | `m-c` (machine, calculated) | — |
| `time` | when the message was formed | ISO-8601 UTC, 2 decimal places |
| `start` | when the emission began | ISO-8601 UTC |
| `stale` | emission end + hold | ISO-8601 UTC |

`<point>`:

| Attribute | Value | Unit |
|---|---|---|
| `lat`, `lon` | **the receiver's position**, not the emitter's | degrees, WGS-84 |
| `hae` | receiver height above the ellipsoid | metres |
| `ce` | radius of the 95 % credible range ring, or `9999999.0` | metres |
| `le` | always `9999999.0` — no height is estimated | metres |

`<detail>`:

| Element | Content |
|---|---|
| `<contact callsign>` | `PMR8/114.8Hz`, matching the order of battle's label |
| `<remarks>` | frequency, channel, duration, SNR, peak power, deviation, tone, range statement, and the bearing caveat |
| `<__esm446>` | the measurement, machine-readable: `frequency_hz`, `pmr_channel`, `snr_db`, `peak_power_dbfs`, `peak_deviation_hz`, `ctcss_tone_hz`, `duration_s`, `calibrated`, `attribution` |

**UID stability is deliberate.** It is derived from what identifies the emitter — channel and
sub-audible tone — not from the emission. Successive transmissions from one radio update one
track instead of littering the map with a new pin every time somebody keys up.

### 3.2 Range rings

One `u-d-r` event per credible percentile (5, 50, 68, 90, 95), emitted **only** when a range
estimate exists. `ce` carries that percentile's radius in metres; `<remarks>` names the
percentile; `<link>` points at the parent track's UID with `relation="p-p"`; `<color argb>`
is set per percentile so tighter rings draw hotter.

No estimate means no rings at all. A single ring with no percentile attached would be read as
certainty, so the meaning travels with the drawing or the drawing is not sent.

---

## 4. Geometry: what the position means

**The emitter's position is not measured and is not transmitted.**

A single omnidirectional antenna measures how strongly a signal arrives. That inverts to a
range, and to nothing else — no bearing, therefore no position. The correct product is an
annulus about the receiver.

This is encoded the way CoT provides for: the point is the **receiver's** position, and `ce`
is the radius within which the emitter lies. A TAK client renders that as an accuracy circle.
A consumer must therefore treat the coordinates as *the sensor's location*, and the emitter as
being somewhere on the disc of radius `ce` around it.

`ce = 9999999.0` is CoT's sentinel for unknown and is what every message currently carries,
because absolute power requires a conducted calibration and none exists
([#41](https://github.com/alesan121/esm446/issues/41)). A default of `0` would have rendered a
pinpoint fix on the receiver, which is the worst available answer: precise and wrong.

---

## 5. Identification: what `a-f-G` claims

`a-f-G` is emitted when the emission carried the configured pre-shared sub-audible CTCSS tone;
`a-u-G` otherwise.

This is **cooperative identification by a pre-shared key**. There is no challenge, no response,
and no cryptography. It is not IFF and not Mode 5. Anybody who knows the tone — which is
selectable from a table of 38 on any handset — produces `a-f-G`. The affiliation states that
the expected code was heard, and nothing beyond that.

`a-u-G` is *unknown*, never *hostile*. An ESM node observes emissions; it has no basis for
declaring intent, and the CoT type hierarchy has a value for exactly this situation.

---

## 6. Cadence, staleness, and loss

| Property | Value | Rationale |
|---|---|---|
| **Trigger** | one message set per emission, as the emission completes | the node has nothing to report until an emission ends and can be measured |
| **Rate** | bounded by traffic on the band; a busy PMR446 channel produces a few per minute | no periodic heartbeat is sent |
| **`stale`** | emission end + `COT_STALE_S`, default **300 s** | a policy, not a measurement: an emitter that stopped transmitting has not stopped existing, but it stops being current |
| **Retransmission** | none | the next transmission from that emitter refreshes the track under the same UID |

**On loss.** A failed send is logged and the message is dropped. Nothing is queued: a buffer
that grows while a link is down eventually takes the process with it, and the capture matters
more than the feed. Stream transports reconnect on exponential backoff from 1 s to 30 s, so a
restarted TAK server is not hammered. The node keeps processing throughout — **losing the feed
must never cost the capture**, which is the same rule the metadata sinks follow.

A consumer that sees a track go quiet cannot distinguish a dead link from a silent band. If
that distinction matters operationally, a heartbeat is needed; there is not one, and this
document says so rather than leaving it to be discovered.

---

## 7. What a consumer must not conclude

1. The coordinates are **not** the emitter's position. See §4.
2. `a-f-G` is **not** an IFF interrogation result. See §5.
3. A range ring drawn from an uncalibrated estimate is model output, not measurement. The
   remarks say `RANGE UNCALIBRATED` and `<__esm446 calibrated="false">` carries it
   machine-readably.
4. Absence of a track is **not** evidence of an absence of emissions. It may be a dropped
   link, an emission below the detection threshold, or one shorter than the tracker's minimum.
5. A detection marked with an `attribution` is a by-product of another emitter's transmission
   — splatter or intermodulation — not an independent emitter. See `esm446.analysis.artefacts`.

---

## 8. Verification

| Requirement | How it is verified | Test |
|---|---|---|
| Every event validates against the CoT schema | XSD validation of generated messages | `test_every_event_validates_against_the_schema` |
| The message does not depend on the transport | byte comparison across UDP, TCP and TLS | `test_the_same_emission_is_identical_over_every_transport` |
| Uncalibrated ranges are marked | inspection of remarks and `ce` | `test_an_uncalibrated_emission_carries_an_unknown_circular_error` |
| A stable UID per emitter | two emissions, one UID | `test_two_transmissions_from_one_emitter_share_a_uid` |
| Loss does not stop the capture | publish with no listener | `test_a_dead_link_does_not_raise` |
| `stale` follows the stated policy | arithmetic against the emission | `test_stale_is_the_end_of_the_emission_plus_the_hold` |

---

## 9. Configuration

| Setting | Default | Meaning |
|---|---|---|
| `ESM446_COT_DESTINATION` | *(none)* | transport URL; nothing published when unset |
| `ESM446_COT_LATITUDE` | `0.0` | receiver latitude, degrees north |
| `ESM446_COT_LONGITUDE` | `0.0` | receiver longitude, degrees east |
| `ESM446_COT_ALTITUDE_M` | `0.0` | receiver height above the ellipsoid, metres |
| `ESM446_COT_CALLSIGN` | `ESM-446` | name the receiver appears under |
| `ESM446_COT_STALE_S` | `300.0` | hold after an emission ends before its track goes stale |

Publishing with the position left at its default logs a warning: every track would otherwise
land at 0° N 0° E.

---

## 10. Example

An emission on PMR446 channel 8 carrying the pre-shared 114.8 Hz tone, with no calibration:

```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<event version="2.0" uid="ESM446.PMR8.114.8" type="a-f-G" how="m-c"
       time="2026-08-13T22:14:06.43Z" start="2026-08-13T22:14:02.25Z"
       stale="2026-08-13T22:19:06.43Z">
  <point lat="40.4168000" lon="-3.7038000" hae="0.0" ce="9999999.0" le="9999999.0"/>
  <detail>
    <contact callsign="PMR8/114.8Hz"/>
    <remarks>446.09376 MHz | PMR8 | 4.2 s | SNR 40.9 dB | peak -0.6 dBFS | dev 1347 Hz |
             CTCSS 114.8 Hz | RANGE UNKNOWN: power is uncalibrated |
             bearing not measured: single omnidirectional sensor</remarks>
    <__esm446 frequency_hz="446093757.0" pmr_channel="8" snr_db="40.90"
              peak_power_dbfs="-0.60" peak_deviation_hz="1347.0" ctcss_tone_hz="114.8"
              duration_s="4.180" calibrated="false" attribution=""/>
  </detail>
</event>
```

Note `ce="9999999.0"` and `RANGE UNKNOWN`. The system has detected an emitter, identified its
channel and its cooperative code, measured its deviation — and states plainly that it does not
know how far away it is.
