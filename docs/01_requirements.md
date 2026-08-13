# System requirements — ESM-446

**Document:** REQ-ESM446
**Version:** 1.0
**Status:** baselined against the implementation at the time of writing

Numbered requirements with a verification method and, for every one of them, the test that
verifies it. The verification column follows the usual four: **A**nalysis, **T**est,
**I**nspection, **D**emonstration.

The `Verified by` column is not decoration. `tests/test_requirements.py` parses this file and
fails the build if any requirement has no verifying test, or names a test that does not exist.
A requirements document that drifts from the code is worse than none, because it is believed.

**Status values.** `MET` — verified, passing. `PARTIAL` — verified to the extent the available
equipment allows, with the shortfall stated. `BLOCKED` — cannot be verified with what is
available, with the blocker named. Nothing is marked met on the strength of an intention.

---

## 1. Functional requirements

| ID | Requirement | Method | Verified by | Status |
|---|---|---|---|---|
| REQ-FUN-001 | The node shall split the sampled band into uniformly spaced channels matching the 12.5 kHz PMR446 raster. | T | `test_channel_bin_mapping_agrees_with_python` | MET |
| REQ-FUN-002 | Each channel filter shall reject an emission on an adjacent channel by at least 60 dB. | T | `test_adjacent_channel_rejection` | MET |
| REQ-FUN-003 | The node shall detect emissions at a stated, configurable probability of false alarm, independent of the noise level. | T | `test_false_alarm_rate_holds_across_noise_levels`, `test_pure_receiver_noise_produces_no_false_alarms`, `test_tracking_the_level_cuts_the_rate_on_ambient_noise` | PARTIAL — design 1e-8. Met on synthetic noise across nine orders of magnitude, and no crossing at all measured on real receiver noise; on a real band at high gain the rate is 8e-6, because the environment is neither Gaussian nor stationary |
| REQ-FUN-004 | The node shall report the sensitivity penalty for an emitter offset from a channel centre. | A, T | `test_the_worst_case_is_six_decibels_at_the_bin_edge` | MET |
| REQ-FUN-005 | The node shall identify the sub-audible CTCSS tone of an emission from the 38-tone standard table. | T | `test_identifies_every_kind_of_tone_through_the_full_chain` | MET |
| REQ-FUN-006 | The node shall classify an emission as FRIEND only when its tone matches the configured pre-shared code, and UNKNOWN otherwise. | T | `test_the_tones_distinguish_friend_from_unknown` | MET |
| REQ-FUN-007 | The node shall report emissions that fall outside the nominal PMR446 channel plan as off-grid rather than snapping them to the nearest channel. | T | `test_demo_reports_the_off_grid_emitter_as_off_grid` | MET |
| REQ-FUN-008 | The node shall measure the peak frequency deviation of each emission. | T | `test_deviation_is_within_the_etsi_limit` | MET |
| REQ-FUN-009 | The node shall record the receiver gain configuration with every emission. | T | `test_sqlite_lifts_the_gains_into_columns` | MET |
| REQ-FUN-010 | The node shall persist emission metadata such that a process killed mid-capture retains everything written before the failure. | T | `test_jsonl_survives_a_truncated_last_line` | MET |
| REQ-FUN-011 | The system shall aggregate stored emissions into an order of battle: distinct emitters, channel occupancy by hour, and burst statistics. | T | `test_the_cli_renders_an_order_of_battle_from_a_store` | MET |
| REQ-FUN-012 | The system shall report an emitter count as a lower bound unless two of that emitter's emissions overlap in time. | T | `test_a_count_is_a_lower_bound_unless_transmissions_overlap` | MET |
| REQ-FUN-013 | The system shall attribute a detection to a stronger simultaneous emission where an exact frequency relation identifies it as that emission's by-product. | T | `test_attribution_reduces_them_to_the_two_that_transmitted` | MET |
| REQ-FUN-014 | The system shall never delete a detection it has attributed as a by-product. | T | `test_nothing_is_ever_removed` | MET |
| REQ-FUN-015 | The node shall exclude the receiver's own local-oscillator leakage from detection. | T | `test_node_ignores_the_dc_bin` | MET |
| REQ-FUN-016 | The system shall estimate range from received power with uncertainty propagated by Monte Carlo, reporting empirical percentiles. | T | `test_the_interval_is_skewed_not_symmetric` | MET |
| REQ-FUN-017 | The system shall run the same pipeline over live capture and over recorded IQ, with no code path exclusive to either. | T, D | `test_simulated_file_replays_through_the_node` | MET |

## 2. Performance requirements

| ID | Requirement | Method | Verified by | Status |
|---|---|---|---|---|
| REQ-PER-001 | The node shall process 2 MS/s across 160 channels in real time with a margin of at least 2x on the development machine. | T | `test_main_succeeds_within_a_generous_budget` | MET — median of five runs 0.26 CPU-s/s, 3.9x |
| REQ-PER-002 | The channeliser shall cost no more than 0.5 CPU-seconds per signal second. | T | `test_pfb_benchmark_keeps_up_with_real_time` | MET — median of five runs 0.20 |
| REQ-PER-003 | The wideband survey shall cost no more than 5 % of the channeliser. | T | `test_analyse_does_not_hold_the_whole_spectrogram` | MET — measured 0.009 |
| REQ-PER-004 | Memory use shall not scale with capture length. | T | `test_waterfall_is_capped_rather_than_allocating_gigabytes` | MET |
| REQ-PER-005 | The false alarm rate shall not change when the noise estimate is held between frames. | T | `test_holding_the_noise_estimate_does_not_change_the_false_alarm_rate` | MET |

## 3. Interface requirements

| ID | Requirement | Method | Verified by | Status |
|---|---|---|---|---|
| REQ-IF-001 | Emissions shall be published as Cursor-on-Target events valid against the CoT schema. | T | `test_every_event_validates_against_the_schema` | MET |
| REQ-IF-002 | The CoT message shall be identical regardless of the transport carrying it. | T | `test_the_same_emission_is_identical_over_every_transport` | MET |
| REQ-IF-003 | The transport shall be selectable between UDP, TCP and TLS by configuration. | T | `test_open_transport_selects_by_scheme` | MET |
| REQ-IF-004 | A failed publication shall not interrupt capture. | T | `test_a_dead_link_does_not_raise` | MET |
| REQ-IF-005 | The CoT event shall place the emitter at the receiver's position with the circular error carrying the range uncertainty, and shall not assert a bearing. | I, T | `test_the_position_is_the_receivers_not_the_emitters` | MET |
| REQ-IF-006 | An emission with no calibrated range shall carry CoT's unknown-error sentinel, never a default. | T | `test_an_uncalibrated_emission_carries_an_unknown_circular_error` | MET |
| REQ-IF-007 | Successive transmissions from one emitter shall update one track. | T | `test_two_transmissions_from_one_emitter_share_a_uid` | MET |
| REQ-IF-008 | Stored emissions shall be readable back into the same records that were written. | T | `test_reports_survive_a_round_trip_through_sqlite` | MET |
| REQ-IF-009 | A store written by an earlier version shall remain readable and writable. | T | `test_an_archive_from_an_earlier_version_is_brought_up_to_date` | MET |

## 4. Calibration and measurement requirements

| ID | Requirement | Method | Verified by | Status |
|---|---|---|---|---|
| REQ-CAL-001 | Absolute received power shall be reported only where a calibration exists for that exact gain configuration and level range. | T | `test_power_is_not_reported_in_dbm` | MET |
| REQ-CAL-002 | An uncalibrated power estimate shall be marked uncalibrated everywhere it appears. | T | `test_an_uncalibrated_estimate_is_flagged_all_the_way_through` | MET |
| REQ-CAL-003 | The system shall refuse to produce a range estimate from an uncalibrated power reading. | T | `test_an_uncalibrated_report_produces_no_range` | MET |
| REQ-CAL-004 | The receiver shall be characterised for absolute power against a known source. | T | — | **BLOCKED** — needs a calibrated source; see [#41](https://github.com/alesan121/esm446/issues/41) |
| REQ-CAL-005 | Credible intervals shall contain the truth at the frequency they declare. | T | `test_the_ninety_five_percent_ring_contains_the_truth_that_often`, `test_the_rings_undercover_badly_when_the_environment_is_clearer` | PARTIAL — 95.3 % achieved against the model's own prior; the prior is unvalidated and needs REQ-CAL-004. Sensitivity to that now measured: coverage holds when the environment is more obstructed than assumed and falls to 34 % at an exponent of 2.5 |
| REQ-CAL-006 | Frequencies reported shall be consistent across captures to within 100 Hz. | T | `test_a_drifting_handset_is_still_one_emitter`, `test_a_centred_multiplex_is_measured_where_it_is` | PARTIAL — consistency verified. Absolute accuracy still unmeasured: the measurement is implemented and tested (`esm446-calibrate-frequency`) but needs a GPS-locked reference, and no television multiplex is receivable indoors on a whip cut for 446 MHz. See §9 of the V&V report |

## 5. Legal and ethical requirements

| ID | Requirement | Method | Verified by | Status |
|---|---|---|---|---|
| REQ-LEG-001 | The system shall record signal metadata only, never communication content. | T, I | `test_jsonl_never_stores_audio` | MET |
| REQ-LEG-002 | Demodulated audio shall not leave the function that identifies the sub-audible tone. | I | `test_audio_recording_is_off_by_default` | MET |
| REQ-LEG-003 | Recorded test vectors shall contain only the operator's own transmissions. | I | `test_the_vector_is_small_enough_to_commit` | MET — see `docs/06_legal_ethics.md` |
| REQ-LEG-004 | The node shall publish to a network only where a destination has been configured. | T | `test_no_destination_publishes_nowhere` | MET |
| REQ-LEG-005 | Network services shall bind to loopback unless told otherwise. | I, T | `test_the_server_publishes_nothing_with_nobody_listening` | MET |

## 6. Configuration requirements

| ID | Requirement | Method | Verified by | Status |
|---|---|---|---|---|
| REQ-CFG-001 | A receiver configuration that would mistune the channel plan shall be rejected at startup, not at runtime. | T | `test_settings_rejects_the_band_midpoint_as_centre_frequency` | MET |
| REQ-CFG-002 | A centre frequency on a nominal PMR446 channel shall be rejected, because the receiver's own spur would sit on it. | T | `test_settings_rejects_every_nominal_channel_as_a_centre` | MET |
| REQ-CFG-003 | A sample rate the hardware cannot produce shall be rejected. | T | `test_settings_rejects_a_channel_spacing_that_is_not_the_pmr446_step` | MET |

---

## 7. Requirements deliberately not levied

Stating what was not required is part of specifying a system, and each of these was
considered and declined for a reason that is worth recording.

**Bearing or position of an emitter.** One omnidirectional antenna measures range and nothing
else. A requirement to report position would have been met only by inventing one.

**Real-time demodulated audio output.** Excluded by REQ-LEG-001, and the exclusion is
structural rather than a policy setting: the audio never leaves the identification function.

**A specific-emitter-identification capability.** The strongest candidate feature was measured
and takes the same value on two different radio models — see `docs/04_link_budget.md`. A
requirement would have been unverifiable with the equipment available.

**A detection heartbeat on the CoT interface.** A consumer therefore cannot distinguish a
dead link from a quiet band. Declared in `docs/03_icd_cot.md` §6 rather than left to be
discovered.
