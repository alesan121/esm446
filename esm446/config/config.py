"""Runtime configuration for the ESM-446 node.

Every tunable the node has lives here, read from the environment through
`pydantic_settings.BaseSettings` and validated at import time. Nothing else in the codebase
calls `os.getenv`, and no module instantiates `Settings` itself: they import the
module-level `settings` singleton defined as the last statement of this file.

Validating at import time is deliberate. A receiver misconfiguration -- a sample rate the
hardware cannot produce, a channel count that puts the band plan half a bin off the grid --
is not something to discover after an hour of capture. It should stop the process before
the SDR is ever opened, which is what the validators below do.
"""

from __future__ import annotations

import logging

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

#: Application version. Managed by python-semantic-release; do not edit by hand.
APP_VERSION = "0.0.0-dev"

#: Centre frequency of PMR446 channel 1 (Hz). Duplicated from `esm446.core.bands` because
#: configuration must not import the signal-processing package at settings-validation time.
_PMR446_CHANNEL_1_HZ = 446_006_250

#: Channel spacing of the analogue PMR446 allocation (Hz).
_PMR446_SPACING_HZ = 12_500

#: Number of analogue PMR446 channels.
_PMR446_CHANNEL_COUNT = 16

#: Offset-tuned receiver centre frequency (Hz). Mirrors `esm446.core.bands.DEFAULT_CENTRE_HZ`.
_DEFAULT_CENTRE_HZ = 446_593_750


class Settings(BaseSettings):
    """Node configuration, populated from the environment or a `.env` file.

    Attributes:
        APP_VERSION: Application version string, mirrored from the module constant.
        LOG_LEVEL: Root logging level applied at startup.
        SDR_DRIVER: SoapySDR driver name used to open the receiver.
        SDR_SAMPLE_RATE_HZ: Receiver sample rate. Must be one the device supports.
        SDR_CENTRE_FREQ_HZ: Receiver centre frequency. Must sit on the 12.5 kHz grid.
        SDR_LNA_GAIN_DB: HackRF internal LNA gain, quantised to 8 dB steps by the driver.
        SDR_VGA_GAIN_DB: HackRF baseband VGA gain, quantised to 2 dB steps by the driver.
        SDR_AMP_ENABLED: Whether the HackRF front-end amplifier is switched in.
        EXTERNAL_GAIN_DB: Gain of everything ahead of the SDR, such as an external LNA.
        CHANNELIZER_NUM_CHANNELS: Number of polyphase filter bank channels.
        CHANNELIZER_DECIMATION: Input samples consumed per output frame.
        CHANNELIZER_TAPS_PER_PHASE: Prototype filter length in units of the channel count.
        CFAR_PFA: Design probability of false alarm per bin per frame.
        CFAR_METHOD: CFAR estimator, `ca` for cell averaging or `os` for order statistic.
        CTCSS_EXPECTED_TONE_HZ: Pre-shared sub-audible tone treated as cooperative
            identification, or `None` when no tone is configured.
        CALIBRATION_PATH: Path to the YAML power calibration file.
        RECORD_AUDIO: Whether demodulated audio may be written to disk. Off by default;
            see `docs/06_legal_ethics.md` before switching it on.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ESM446_",
        extra="ignore",
    )

    APP_VERSION: str = APP_VERSION
    LOG_LEVEL: str = "INFO"

    SDR_DRIVER: str = "hackrf"
    SDR_SAMPLE_RATE_HZ: int = 2_000_000
    SDR_CENTRE_FREQ_HZ: int = 446_593_750
    SDR_LNA_GAIN_DB: int = 32
    SDR_VGA_GAIN_DB: int = 20
    SDR_AMP_ENABLED: bool = False
    EXTERNAL_GAIN_DB: float = 20.0

    CHANNELIZER_NUM_CHANNELS: int = 160
    CHANNELIZER_DECIMATION: int = 80
    CHANNELIZER_TAPS_PER_PHASE: int = 12

    CFAR_PFA: float = Field(default=1e-8, gt=0.0, lt=1.0)
    CFAR_METHOD: str = "os"

    CTCSS_EXPECTED_TONE_HZ: float | None = None

    CALIBRATION_PATH: str = "config/calibration.yaml"
    RECORD_AUDIO: bool = False

    @field_validator("CFAR_METHOD")
    @classmethod
    def _validate_cfar_method(cls, value: str) -> str:
        """Reject an unknown CFAR estimator name.

        Args:
            value: Requested estimator name.

        Returns:
            The validated estimator name.
        """
        if value not in ("ca", "os"):
            raise ValueError(f"CFAR_METHOD must be 'ca' or 'os', got {value!r}")
        return value

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Reject a log level the logging module does not recognise.

        Args:
            value: Requested log level name.

        Returns:
            The validated level name, upper-cased.
        """
        level = value.upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"LOG_LEVEL {value!r} is not a recognised logging level")
        return level

    @model_validator(mode="after")
    def _validate_channeliser_geometry(self) -> Settings:
        """Reject a receiver configuration that would mistune the band plan.

        Three ways to get this wrong, all of which produce a running system that quietly
        looks at the wrong frequencies, which is why they are refused at startup rather
        than left to be noticed in the data:

        - a decimation that does not divide the channel count, which the filter bank
          cannot fold;
        - a channel spacing that is not the 12.5 kHz PMR446 step;
        - a centre frequency that is not an integer number of steps from channel 1, which
          puts every channel half a bin off its own centre.

        Returns:
            The validated settings instance.
        """
        if self.CHANNELIZER_NUM_CHANNELS % self.CHANNELIZER_DECIMATION != 0:
            raise ValueError(
                f"CHANNELIZER_DECIMATION {self.CHANNELIZER_DECIMATION} must divide "
                f"CHANNELIZER_NUM_CHANNELS {self.CHANNELIZER_NUM_CHANNELS}"
            )

        spacing = self.SDR_SAMPLE_RATE_HZ / self.CHANNELIZER_NUM_CHANNELS
        if abs(spacing - _PMR446_SPACING_HZ) > 1e-6:
            raise ValueError(
                f"SDR_SAMPLE_RATE_HZ / CHANNELIZER_NUM_CHANNELS gives {spacing:.3f} Hz "
                f"per channel; PMR446 requires {_PMR446_SPACING_HZ} Hz"
            )

        steps = (self.SDR_CENTRE_FREQ_HZ - _PMR446_CHANNEL_1_HZ) / _PMR446_SPACING_HZ
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                f"SDR_CENTRE_FREQ_HZ {self.SDR_CENTRE_FREQ_HZ} is {steps:.4f} channel "
                f"steps from PMR446 channel 1, not an integer; every channel would land "
                f"off its bin centre. Use {_DEFAULT_CENTRE_HZ} Hz."
            )

        # A direct-conversion receiver leaks its local oscillator into the mixer, so a spur
        # sits at the centre frequency permanently -- measured at 31 dB above the noise floor
        # on a HackRF One. Tuning to a nominal channel therefore guarantees a phantom emitter
        # on it, and additionally mirrors every channel onto another through IQ imbalance.
        if 0 <= round(steps) < _PMR446_CHANNEL_COUNT:
            raise ValueError(
                f"SDR_CENTRE_FREQ_HZ {self.SDR_CENTRE_FREQ_HZ} is PMR446 channel "
                f"{round(steps) + 1}. The receiver's own DC spur would sit on it. "
                f"Offset-tune instead: {_DEFAULT_CENTRE_HZ} Hz stays on the 12.5 kHz grid "
                f"but places the spur outside the allocation."
            )

        logger.debug(
            "config: validated receiver geometry, %d channels of %.1f Hz at %.6f MHz",
            self.CHANNELIZER_NUM_CHANNELS,
            spacing,
            self.SDR_CENTRE_FREQ_HZ / 1e6,
        )
        return self


settings = Settings()
