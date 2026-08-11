"""Verification of the runtime configuration and its startup validators.

The validators exist to turn a class of silent failure into a startup crash: a receiver
configuration that runs happily while looking at the wrong frequencies. These tests
exercise each way of getting that wrong.

Following the project testing convention, no settings-dependent module is imported at
module level -- `Settings` validates at import time, so a top-level import would pick up
whatever environment the test runner happens to have.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest
from pydantic import ValidationError

_VALID_ENV = {
    "ESM446_SDR_SAMPLE_RATE_HZ": "2000000",
    "ESM446_SDR_CENTRE_FREQ_HZ": "446093750",
    "ESM446_CHANNELIZER_NUM_CHANNELS": "160",
    "ESM446_CHANNELIZER_DECIMATION": "80",
}


@pytest.fixture()
def settings_class():
    """Expose the `Settings` class with a known-valid environment already loaded."""
    with patch.dict("os.environ", _VALID_ENV):
        import esm446.config.config as config_module

        importlib.reload(config_module)
        return config_module.Settings


def test_settings_accepts_the_default_receiver_geometry(settings_class) -> None:
    with patch.dict("os.environ", _VALID_ENV):
        settings = settings_class()

    assert settings.SDR_CENTRE_FREQ_HZ == 446_093_750
    assert settings.SDR_SAMPLE_RATE_HZ / settings.CHANNELIZER_NUM_CHANNELS == 12_500.0


def test_settings_rejects_decimation_that_does_not_divide_channel_count(settings_class) -> None:
    with patch.dict("os.environ", {**_VALID_ENV, "ESM446_CHANNELIZER_DECIMATION": "7"}):
        with pytest.raises(ValidationError, match="must divide"):
            settings_class()


def test_settings_rejects_a_channel_spacing_that_is_not_the_pmr446_step(settings_class) -> None:
    """A sample rate and channel count that do not produce 12.5 kHz bins.

    The sample rate is what varies here, not the channel count: 80 must still divide the
    channel count or the earlier decimation validator fires first and this one is never
    reached.
    """
    with patch.dict("os.environ", {**_VALID_ENV, "ESM446_SDR_SAMPLE_RATE_HZ": "2400000"}):
        with pytest.raises(ValidationError, match="PMR446 requires"):
            settings_class()


def test_settings_rejects_the_band_midpoint_as_centre_frequency(settings_class) -> None:
    """446.1 MHz is 7.5 channel steps from channel 1 -- half a bin off the grid.

    This is the specific mistake that looks most reasonable: it really is the midpoint of
    the allocation. It is also the one that puts every channel across two bins.
    """
    with patch.dict("os.environ", {**_VALID_ENV, "ESM446_SDR_CENTRE_FREQ_HZ": "446100000"}):
        with pytest.raises(ValidationError, match="not an integer"):
            settings_class()


def test_settings_rejects_the_v0_centre_frequency(settings_class) -> None:
    """v0 tuned 446.0935 MHz, 250 Hz off channel 8. The validator catches it."""
    with patch.dict("os.environ", {**_VALID_ENV, "ESM446_SDR_CENTRE_FREQ_HZ": "446093500"}):
        with pytest.raises(ValidationError, match="not an integer"):
            settings_class()


def test_settings_rejects_an_unknown_cfar_method(settings_class) -> None:
    with patch.dict("os.environ", {**_VALID_ENV, "ESM446_CFAR_METHOD": "magic"}):
        with pytest.raises(ValidationError, match="'ca' or 'os'"):
            settings_class()


@pytest.mark.parametrize("pfa", ["0", "1", "1.5", "-0.1"])
def test_settings_rejects_a_pfa_outside_the_unit_interval(settings_class, pfa: str) -> None:
    with patch.dict("os.environ", {**_VALID_ENV, "ESM446_CFAR_PFA": pfa}):
        with pytest.raises(ValidationError):
            settings_class()


def test_settings_rejects_an_unknown_log_level(settings_class) -> None:
    with patch.dict("os.environ", {**_VALID_ENV, "ESM446_LOG_LEVEL": "CHATTY"}):
        with pytest.raises(ValidationError, match="not a recognised logging level"):
            settings_class()


def test_settings_normalises_log_level_case(settings_class) -> None:
    with patch.dict("os.environ", {**_VALID_ENV, "ESM446_LOG_LEVEL": "debug"}):
        assert settings_class().LOG_LEVEL == "DEBUG"


def test_audio_recording_is_off_by_default(settings_class) -> None:
    """Metadata-only is the default posture, not something the operator opts into."""
    with patch.dict("os.environ", _VALID_ENV):
        assert settings_class().RECORD_AUDIO is False


def test_no_ctcss_tone_is_configured_by_default(settings_class) -> None:
    """Without a pre-shared tone every emitter is UNKNOWN, which is the honest default."""
    with patch.dict("os.environ", _VALID_ENV):
        assert settings_class().CTCSS_EXPECTED_TONE_HZ is None


def test_module_exposes_a_settings_singleton(settings_class) -> None:
    """Every module imports this instance; nothing instantiates `Settings` itself."""
    with patch.dict("os.environ", _VALID_ENV):
        import esm446.config.config as config_module

        importlib.reload(config_module)
        assert isinstance(config_module.settings, config_module.Settings)


def test_app_version_constant_matches_the_settings_default(settings_class) -> None:
    """python-semantic-release rewrites APP_VERSION; the default must track it."""
    with patch.dict("os.environ", _VALID_ENV):
        import esm446.config.config as config_module

        importlib.reload(config_module)
        assert config_module.settings.APP_VERSION == config_module.APP_VERSION
