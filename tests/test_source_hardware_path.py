"""Verification of the live-capture path, which no test had ever executed.

`SoapySource` is the code that runs against the HackRF, and it was the only substantial
module in the project with no coverage: it imports SoapySDR inside its constructor, so
without the SDR stack installed nothing could reach it. That is exactly backwards — the code
that talks to hardware is the code whose mistakes are most expensive and hardest to notice,
and both of the defects it was written to fix were silent ones in v0.

A stand-in for SoapySDR is installed into `sys.modules` and records every call. That verifies
the two things a real radio could not tell us any better: that the arguments are in the right
positions, and that the gains the device reports back are the ones written down. What it
cannot verify is that the driver behaves as its documentation says; that needs the radio.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

SOAPY_SDR_RX = 1
SOAPY_SDR_CF32 = "CF32"


@dataclass
class FakeStatus:
    """What `readStream` returns: a count, or a negative error code."""

    ret: int
    flags: int = 0


@dataclass
class FakeDevice:
    """A SoapySDR device that records what was asked of it."""

    args: dict[str, str]
    sample_rates: list[float] = field(default_factory=lambda: [2e6, 8e6, 10e6, 20e6])
    calls: list[tuple] = field(default_factory=list)
    read_returns: int | None = None
    read_flags: int = 0
    samples_per_read: complex = 0.5 + 0.25j

    def listSampleRates(self, direction: int, channel: int) -> list[float]:  # noqa: N802
        self.calls.append(("listSampleRates", direction, channel))
        return self.sample_rates

    def setSampleRate(self, direction: int, channel: int, rate: float) -> None:  # noqa: N802
        self.calls.append(("setSampleRate", direction, channel, rate))

    def setFrequency(self, direction: int, channel: int, hz: float) -> None:  # noqa: N802
        self.calls.append(("setFrequency", direction, channel, hz))

    def setGain(self, direction: int, channel: int, name: str, value: float) -> None:  # noqa: N802
        self.calls.append(("setGain", direction, channel, name, value))

    def setupStream(self, direction: int, fmt: str, channels: list[int]) -> str:  # noqa: N802
        self.calls.append(("setupStream", direction, fmt, tuple(channels)))
        return "stream-handle"

    def activateStream(self, stream: str) -> None:  # noqa: N802
        self.calls.append(("activateStream", stream))

    def readStream(self, stream, buffers, count, timeoutUs):  # noqa: N802, N803
        self.calls.append(("readStream", stream, count, timeoutUs))
        returned = count if self.read_returns is None else self.read_returns
        if returned > 0:
            buffers[0][:returned] = self.samples_per_read
        return FakeStatus(returned, flags=self.read_flags)

    def deactivateStream(self, stream: str) -> None:  # noqa: N802
        self.calls.append(("deactivateStream", stream))

    def closeStream(self, stream: str) -> None:  # noqa: N802
        self.calls.append(("closeStream", stream))

    def named(self, name: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == name]


@pytest.fixture
def soapy(monkeypatch: pytest.MonkeyPatch) -> list[FakeDevice]:
    """Install the stand-in and hand back the list of devices it creates."""
    created: list[FakeDevice] = []

    module = types.ModuleType("SoapySDR")
    module.SOAPY_SDR_RX = SOAPY_SDR_RX
    module.SOAPY_SDR_CF32 = SOAPY_SDR_CF32
    module.SOAPY_SDR_OVERFLOW = 4

    def device(args: dict[str, str]) -> FakeDevice:
        made = FakeDevice(args=args)
        created.append(made)
        return made

    module.Device = device
    monkeypatch.setitem(sys.modules, "SoapySDR", module)
    return created


def open_source(**overrides: Any):
    from esm446.core.source import SoapySource

    settings: dict[str, Any] = {
        "sample_rate": 2_000_000.0,
        "centre_frequency": 446_593_750.0,
        "lna_gain_db": 32.0,
        "vga_gain_db": 20.0,
    }
    settings.update(overrides)
    return SoapySource(**settings)


# --------------------------------------------------------------------------------------
# The two defects this class exists to prevent
# --------------------------------------------------------------------------------------


def test_the_gain_call_puts_the_channel_where_the_channel_goes(soapy) -> None:
    """v0 called setGain(RX, 8, "LNA", value) believing the 8 was a gain.

    The second argument is the channel index, and the HackRF has only channel 0, so no gain
    was ever applied and nobody noticed. This is that bug, made impossible to reintroduce
    without a test failing.
    """
    open_source()
    device = soapy[0]

    for direction, channel, name, value in (c[1:] for c in device.named("setGain")):
        assert direction == SOAPY_SDR_RX
        assert channel == 0, f"the {name} gain went to channel {channel}, not the channel index 0"
        assert isinstance(value, float)


def test_an_unsupported_sample_rate_is_refused_with_the_supported_ones(soapy) -> None:
    """v0 asked for 800 kS/s, below the HackRF's minimum, and the driver gave it something
    else -- so the entire channel grid was mistuned and the output was confidently wrong."""
    with pytest.raises(ValueError, match=r"does not support 0.800 MS/s"):
        open_source(sample_rate=800_000.0)


def test_the_supported_rates_are_named_in_the_refusal(soapy) -> None:
    """An error that says no without saying what would work costs somebody an afternoon."""
    with pytest.raises(ValueError, match="2.0"):
        open_source(sample_rate=800_000.0)


# --------------------------------------------------------------------------------------
# Configuration actually reaching the device
# --------------------------------------------------------------------------------------


def test_the_rate_and_frequency_are_set_before_the_stream_is_opened(soapy) -> None:
    """Order matters: configuring a running stream is not reliable across drivers."""
    open_source()
    order = [c[0] for c in soapy[0].calls]

    assert order.index("setSampleRate") < order.index("setupStream")
    assert order.index("setFrequency") < order.index("setupStream")
    assert order.index("activateStream") > order.index("setupStream")


def test_the_requested_gains_are_quantised_to_what_the_hardware_can_do(soapy) -> None:
    """The HackRF steps the LNA by 8 dB and the VGA by 2 dB. Asking for 35 gets 32."""
    source = open_source(lna_gain_db=35.0, vga_gain_db=21.0)
    applied = {c[3]: c[4] for c in soapy[0].named("setGain")}

    assert applied["LNA"] == 32.0
    assert applied["VGA"] == 20.0
    assert source.gains.lna_db == 32.0
    assert source.gains.vga_db == 20.0


def test_the_gains_recorded_are_the_ones_applied(soapy) -> None:
    """Every emission carries these, and a calibration is worthless if they are aspirations."""
    source = open_source(lna_gain_db=40.0, vga_gain_db=62.0, amp_enabled=True)
    applied = {c[3]: c[4] for c in soapy[0].named("setGain")}

    assert applied["LNA"] == source.gains.lna_db
    assert applied["VGA"] == source.gains.vga_db
    assert applied["AMP"] == 14.0
    assert source.gains.amp_enabled is True


def test_the_amplifier_stays_off_unless_asked_for(soapy) -> None:
    open_source(amp_enabled=False)
    applied = {c[3]: c[4] for c in soapy[0].named("setGain")}

    assert applied["AMP"] == 0.0


def test_the_driver_is_selectable(soapy) -> None:
    open_source(driver="rtlsdr")

    assert soapy[0].args == {"driver": "rtlsdr"}


def test_a_device_reporting_no_rate_list_is_accepted(soapy, monkeypatch) -> None:
    """Some drivers report a continuous range rather than a list. Refusing them would be
    wrong: the check exists to catch an unsupported rate, not to require an enumeration."""
    from esm446.core.source import SoapySource

    original = FakeDevice.listSampleRates
    monkeypatch.setattr(FakeDevice, "listSampleRates", lambda self, d, c: [])
    source = SoapySource(
        sample_rate=800_000.0, centre_frequency=446_593_750.0, lna_gain_db=0.0, vga_gain_db=0.0
    )
    del original

    assert source.sample_rate == 800_000.0


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def test_a_read_returns_the_samples_the_driver_delivered(soapy) -> None:
    source = open_source()
    block = source.read(4096)

    assert block is not None
    assert block.dtype == np.complex64
    assert len(block) == 4096


def test_a_short_read_is_truncated_to_what_arrived(soapy) -> None:
    """The driver is under no obligation to fill the buffer, and reading past what it wrote
    would feed uninitialised memory into the channeliser."""
    source = open_source()
    soapy[0].read_returns = 100

    block = source.read(4096)

    assert len(block) == 100


def test_a_driver_error_yields_nothing_rather_than_raising(soapy) -> None:
    """A timeout or an overflow must not end the capture: the next read may well succeed."""
    source = open_source()
    soapy[0].read_returns = -1

    block = source.read(4096)

    assert block is not None
    assert len(block) == 0


# --------------------------------------------------------------------------------------
# Sample-loss accounting -- v0's equivalent was `if sr.ret < 0: continue`, silently
# --------------------------------------------------------------------------------------


def test_a_clean_read_records_no_gap(soapy) -> None:
    source = open_source()
    source.read(4096)

    assert source.overflow_count == 0
    assert source.gaps == []


def test_an_overflow_flagged_read_is_counted_even_though_samples_arrived(soapy) -> None:
    """The driver can deliver a full buffer and still flag that a gap sits next to it.

    Silently trusting a full-looking buffer here is exactly the class of bug this exists to
    catch: the data is real, but it is not contiguous with what came before it, and every
    downstream duty-cycle and duration figure needs to know that.
    """
    source = open_source()
    soapy[0].read_flags = 4  # SOAPY_SDR_OVERFLOW

    block = source.read(4096)

    assert len(block) == 4096, "the samples that did arrive must still reach the caller"
    assert source.overflow_count == 1
    assert source.gaps[0]["overflow"] == 1.0


def test_a_driver_error_is_recorded_as_a_gap_too(soapy) -> None:
    source = open_source()
    soapy[0].read_returns = -1

    source.read(4096)

    assert len(source.gaps) == 1
    assert source.gaps[0]["samples_delivered"] == 0.0


def test_a_timeout_with_nothing_delivered_is_a_gap_but_not_an_overflow(soapy) -> None:
    """A timeout is not necessarily loss -- the band can be legitimately quiet -- but the
    node's own accounting still needs to see that no data arrived for this call."""
    source = open_source()
    soapy[0].read_returns = 0

    source.read(4096)

    assert len(source.gaps) == 1
    assert source.overflow_count == 0


def test_gaps_accumulate_across_reads(soapy) -> None:
    source = open_source()
    soapy[0].read_flags = 4

    source.read(4096)
    source.read(4096)
    source.read(4096)

    assert source.overflow_count == 3
    assert len(source.gaps) == 3
    assert source.samples_lost_estimate == 3


def test_live_capture_timestamps_from_the_clock(soapy) -> None:
    """The one case where reading the wall clock is right, and the reason replay does not."""
    import time

    before = time.time()
    source = open_source()

    assert before <= source.start_time <= time.time()


# --------------------------------------------------------------------------------------
# Shutting down
# --------------------------------------------------------------------------------------


def test_closing_deactivates_before_closing_the_stream(soapy) -> None:
    source = open_source()
    source.close()
    order = [c[0] for c in soapy[0].calls]

    assert order.index("deactivateStream") < order.index("closeStream")


def test_closing_twice_is_harmless(soapy) -> None:
    """It happens on every error path that unwinds through a context manager."""
    source = open_source()
    source.close()
    source.close()

    assert len(soapy[0].named("closeStream")) == 1


def test_the_source_works_as_a_context_manager(soapy) -> None:
    from esm446.core.source import SoapySource

    with SoapySource(
        sample_rate=2_000_000.0,
        centre_frequency=446_593_750.0,
        lna_gain_db=0.0,
        vga_gain_db=0.0,
    ):
        pass

    assert soapy[0].named("closeStream")


# --------------------------------------------------------------------------------------
# Replay guards that had no test either
# --------------------------------------------------------------------------------------


def test_a_capture_ending_mid_sample_drops_the_partial_pair(tmp_path) -> None:
    """A truncated recording ends with an I and no Q.

    Keeping it would shift the I/Q phase of everything read afterwards by one component,
    which does not raise anything, does not look wrong, and quietly rotates the whole
    capture.
    """
    from esm446.core.source import FileSource

    path = tmp_path / "truncated.cs8"
    # Nine int8 values: four complete samples and half of a fifth.
    np.arange(9, dtype=np.int8).tofile(path)

    with FileSource(path, 2_000_000.0, 446_593_750.0, "cs8") as source:
        block = source.read(16)

    assert block is not None
    assert len(block) == 4, "the partial pair was kept and every sample after it is rotated"


def test_a_sidecar_gives_the_capture_its_own_start_time(tmp_path) -> None:
    """Without this a replayed recording is stamped with when somebody analysed it."""
    import json

    from esm446.core.source import FileSource

    path = tmp_path / "capture.cs8"
    np.zeros(64, dtype=np.int8).tofile(path)
    path.with_suffix(".json").write_text(json.dumps({"start_time": 1_700_000_000.0}))

    with FileSource(path, 2_000_000.0, 446_593_750.0, "cs8") as source:
        assert source.start_time == 1_700_000_000.0


def test_an_unreadable_sidecar_falls_back_instead_of_failing(tmp_path) -> None:
    """A corrupt sidecar must not cost the capture; the file's own times are good enough."""
    from esm446.core.source import FileSource

    path = tmp_path / "capture.cs8"
    np.zeros(64, dtype=np.int8).tofile(path)
    path.with_suffix(".json").write_text("{not json")

    with FileSource(path, 2_000_000.0, 446_593_750.0, "cs8") as source:
        assert source.start_time > 0
