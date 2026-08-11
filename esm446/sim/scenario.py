"""Generate synthetic PMR446 scenes with ground truth.

Why the simulator carries truth
-------------------------------
A test that puts an emitter on channel 6 and asserts a detection on channel 6 catches gross
errors and nothing else. It cannot say what fraction of emissions were found, how that
degrades with signal-to-noise ratio, or how often the band reports something that was never
transmitted. Those are the questions a detection system is judged on, and answering them
requires the scene and the record of what went into it to be produced together, from one
seed, reproducibly.

That is what makes this more than a test fixture. The same scenarios feed the unit tests, the
throughput benchmark and the demonstration, so a figure quoted in one is the same figure in
the others.

Modelling choices
-----------------
Received power is derived rather than assigned. An emitter is given a transmit power and a
distance, and a log-distance path loss model turns those into an amplitude. Setting
amplitudes directly would be simpler and would quietly decouple the scenario from the link
budget in `esm446.core.rfchain`, which is precisely the coupling worth keeping: if the
predicted detection range and the simulated one disagree, one of them is wrong and it should
be visible.

Four effects are modelled because each one broke something real:

- **Keying ramp.** An instantaneous gate is a step, and a step through an FM discriminator is
  an enormous frequency spike. It read as five times the ETSI deviation limit (#7). Real
  transmitters ramp; the ramp is configurable so both cases can be exercised.
- **Frequency error.** Inexpensive PMR446 handsets sit a few hundred Hz off nominal. An
  emitter parked exactly on a bin centre is the easy case, and the channeliser should not be
  tested only on it.
- **Log-normal shadowing.** Constant over one transmission, redrawn between them, which is
  what an obstruction does as a talker moves.
- **Rayleigh fading.** Multipath, varying within a transmission, which is what puts brief
  nulls in the middle of an over and makes the tracker's hangover necessary.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from esm446.core import bands

logger = logging.getLogger(__name__)

#: Speed of light in m/s.
SPEED_OF_LIGHT = 299_792_458.0

#: Reference distance for the log-distance path loss model, in metres.
REFERENCE_DISTANCE_M = 1.0

#: Nominal peak deviation of a CTCSS tone, in Hz. Typically 10-20% of the channel maximum.
CTCSS_DEVIATION_HZ = 400.0

#: Nominal peak deviation of voice content, in Hz.
VOICE_DEVIATION_HZ = 1_500.0


@dataclass
class Transmission:
    """One over: a single continuous press of the transmit key.

    Attributes:
        start_s: Start time in seconds from the beginning of the scene.
        stop_s: End time in seconds.
    """

    start_s: float
    stop_s: float

    @property
    def duration_s(self) -> float:
        """Length of the transmission in seconds."""
        return self.stop_s - self.start_s


@dataclass
class Emitter:
    """One radio in the scene.

    Attributes:
        name: Identifier carried through to ground truth.
        channel: PMR446 channel number, or ``None`` when ``frequency_hz`` is given directly.
        frequency_hz: Absolute centre frequency. Overrides ``channel`` when set, which is how
            off-grid emitters are placed.
        eirp_dbm: Radiated power. The ETSI EN 300 296 limit is 27 dBm; a handset with a
            2 dBi antenna reaches about 29 dBm EIRP.
        distance_m: Range from the receiver, used with the path loss model.
        ctcss_hz: Sub-audible tone, or ``None`` for an emitter that sends none.
        frequency_error_hz: Offset from nominal, as a real handset would sit.
        deviation_hz: Peak voice deviation.
        ramp_ms: Rise and fall time of the transmit key. Zero gives an instantaneous gate.
        transmissions: The overs this emitter sends.
    """

    name: str
    channel: int | None = None
    frequency_hz: float | None = None
    eirp_dbm: float = 29.0
    distance_m: float = 500.0
    ctcss_hz: float | None = None
    frequency_error_hz: float = 0.0
    deviation_hz: float = VOICE_DEVIATION_HZ
    ramp_ms: float = 5.0
    transmissions: list[Transmission] = field(default_factory=list)

    def centre_frequency(self) -> float:
        """Absolute centre frequency of this emitter, including its frequency error."""
        if self.frequency_hz is not None:
            return self.frequency_hz + self.frequency_error_hz
        if self.channel is None:
            raise ValueError(f"emitter {self.name!r} has neither channel nor frequency_hz")
        return bands.channel_frequency(self.channel) + self.frequency_error_hz


@dataclass
class Propagation:
    """Log-distance path loss with shadowing and fading.

    Attributes:
        path_loss_exponent: 2.0 is free space; 3.5 is the usual urban or wooded value.
        shadowing_sigma_db: Standard deviation of log-normal shadowing.
        rayleigh_fading: Whether to apply multipath fading within a transmission.
    """

    path_loss_exponent: float = 3.5
    shadowing_sigma_db: float = 0.0
    rayleigh_fading: bool = False

    def free_space_loss_db(self, frequency_hz: float) -> float:
        """Path loss at the reference distance, in dB."""
        wavelength = SPEED_OF_LIGHT / frequency_hz
        return 20.0 * math.log10(4.0 * math.pi * REFERENCE_DISTANCE_M / wavelength)

    def path_loss_db(self, frequency_hz: float, distance_m: float) -> float:
        """Total path loss to ``distance_m``, in dB."""
        distance = max(distance_m, REFERENCE_DISTANCE_M)
        return self.free_space_loss_db(frequency_hz) + 10.0 * self.path_loss_exponent * math.log10(
            distance / REFERENCE_DISTANCE_M
        )


@dataclass
class TruthEmission:
    """What the simulator actually put into the scene.

    This is the record detections are scored against, so it holds the parameters a detector
    could in principle recover, and not the internal ones it could not.

    Attributes:
        emitter: Name of the radio that sent it.
        start_s: Start time in seconds from the beginning of the scene.
        stop_s: End time in seconds.
        frequency_hz: Transmitted centre frequency, including the emitter's frequency error.
        pmr_channel: Nearest PMR446 channel, or ``None`` when the emission is off-grid.
        ctcss_hz: Sub-audible tone transmitted, or ``None``.
        received_dbm: Power arriving at the receiver after path loss and shadowing.
        snr_db: Signal-to-noise ratio **in the channel bandwidth**, not across the captured
            band. This is what determines detectability, and it is about 22 dB above the
            wideband figure at 2 MS/s.
    """

    emitter: str
    start_s: float
    stop_s: float
    frequency_hz: float
    pmr_channel: int | None
    ctcss_hz: float | None
    received_dbm: float
    snr_db: float

    @property
    def duration_s(self) -> float:
        """Length of the emission in seconds."""
        return self.stop_s - self.start_s

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary, ready for JSON or YAML."""
        return asdict(self)


@dataclass
class Scenario:
    """A complete synthetic scene.

    Attributes:
        name: Identifier used in reports.
        duration_s: Length of the scene.
        sample_rate: IQ sample rate in Hz.
        centre_frequency: Receiver centre frequency in Hz.
        noise_floor_dbm: Thermal noise power across the whole captured bandwidth.
        emitters: The radios present.
        propagation: Channel model.
        seed: Seed for every random draw, so a scene is reproducible.
    """

    name: str = "scenario"
    duration_s: float = 10.0
    sample_rate: float = 2_000_000.0
    centre_frequency: float = float(bands.DEFAULT_CENTRE_HZ)
    noise_floor_dbm: float = -120.0
    emitters: list[Emitter] = field(default_factory=list)
    propagation: Propagation = field(default_factory=Propagation)
    seed: int = 0

    @property
    def num_samples(self) -> int:
        """Total complex samples in the scene."""
        return int(self.duration_s * self.sample_rate)

    @property
    def processing_gain_db(self) -> float:
        """Gain from concentrating wideband noise into one channel bandwidth.

        The scene's noise is spread across the full captured bandwidth, while a PMR446
        emission occupies one 12.5 kHz channel. Channelising therefore lifts every emitter
        above the noise by the ratio of the two, about 22 dB at 2 MS/s.

        This matters because it is the *in-channel* signal-to-noise ratio that decides
        whether the detector sees an emission. Recording the wideband figure as ground truth
        would make every probability-of-detection curve wrong by 22 dB — a mistake that
        produces a plausible-looking plot.
        """
        return 10.0 * math.log10(self.sample_rate / bands.CHANNEL_SPACING_HZ)

    @property
    def channel_noise_floor_dbm(self) -> float:
        """Noise power within one channel bandwidth."""
        return self.noise_floor_dbm - self.processing_gain_db

    def generate(self) -> tuple[np.ndarray, list[TruthEmission]]:
        """Build the scene.

        Returns:
            ``(iq, truth)`` where ``iq`` is complex64 at ``sample_rate`` and ``truth``
            lists what was transmitted, ordered by start time.
        """
        rng = np.random.default_rng(self.seed)
        total = self.num_samples
        t = np.arange(total) / self.sample_rate

        # Noise amplitude is set so that a signal at noise_floor_dbm lands at 0 dB SNR. All
        # levels in the scene are relative to that reference, which keeps the scenario's
        # dB figures meaningful without pretending to model the receiver's absolute scale.
        noise_amplitude = 10 ** (self.noise_floor_dbm / 20.0)
        scene = (
            noise_amplitude
            * np.sqrt(0.5)
            * (rng.standard_normal(total) + 1j * rng.standard_normal(total))
        ).astype(np.complex64)

        truth: list[TruthEmission] = []
        for emitter in self.emitters:
            truth.extend(self._add_emitter(scene, emitter, t, rng))

        truth.sort(key=lambda e: e.start_s)
        logger.info(
            "scenario: %s, %.1f s, %d emitters, %d transmissions",
            self.name,
            self.duration_s,
            len(self.emitters),
            len(truth),
        )
        return scene, truth

    def _add_emitter(
        self, scene: np.ndarray, emitter: Emitter, t: np.ndarray, rng: np.random.Generator
    ) -> list[TruthEmission]:
        """Add one emitter's transmissions to the scene in place."""
        frequency = emitter.centre_frequency()
        offset = frequency - self.centre_frequency
        if abs(offset) > self.sample_rate / 2:
            raise ValueError(
                f"emitter {emitter.name!r} at {frequency / 1e6:.6f} MHz falls outside the "
                f"{self.sample_rate / 1e6:.1f} MHz captured around "
                f"{self.centre_frequency / 1e6:.6f} MHz"
            )

        loss_db = self.propagation.path_loss_db(frequency, emitter.distance_m)
        records = []

        for transmission in emitter.transmissions:
            start = int(transmission.start_s * self.sample_rate)
            stop = min(int(transmission.stop_s * self.sample_rate), len(scene))
            if stop <= start:
                continue

            # Shadowing is constant within one over and redrawn between them: an obstruction
            # does not change while a talker holds the key down.
            shadowing_db = rng.normal(0.0, self.propagation.shadowing_sigma_db)
            received_dbm = emitter.eirp_dbm - loss_db - shadowing_db
            amplitude = 10 ** (received_dbm / 20.0)

            span = slice(start, stop)
            length = stop - start

            deviation = emitter.deviation_hz * 0.5 * np.sin(2 * np.pi * 640.0 * t[span])
            deviation += emitter.deviation_hz * 0.3 * np.sin(2 * np.pi * 1750.0 * t[span])
            if emitter.ctcss_hz:
                deviation += CTCSS_DEVIATION_HZ * np.sin(2 * np.pi * emitter.ctcss_hz * t[span])

            phase = 2 * np.pi * np.cumsum(deviation) / self.sample_rate
            carrier = np.exp(1j * (2 * np.pi * offset * t[span] + phase))

            envelope = self._keying_envelope(length, emitter.ramp_ms)
            if self.propagation.rayleigh_fading:
                envelope = envelope * self._rayleigh_envelope(length, rng)

            scene[span] += (amplitude * envelope * carrier).astype(np.complex64)

            records.append(
                TruthEmission(
                    emitter=emitter.name,
                    start_s=transmission.start_s,
                    stop_s=stop / self.sample_rate,
                    frequency_hz=frequency,
                    pmr_channel=bands.channel_at(frequency, tolerance_hz=1_000.0),
                    ctcss_hz=emitter.ctcss_hz,
                    received_dbm=received_dbm,
                    snr_db=received_dbm - self.channel_noise_floor_dbm,
                )
            )

        return records

    def _keying_envelope(self, length: int, ramp_ms: float) -> np.ndarray:
        """Raised-cosine rise and fall on the transmit key.

        A hard gate is a step, and a step through an FM discriminator is an enormous
        instantaneous frequency spike that swamps the deviation measurement. Real
        transmitters ramp over a few milliseconds.
        """
        envelope = np.ones(length)
        ramp_samples = int(ramp_ms * 1e-3 * self.sample_rate)
        if ramp_samples < 1 or 2 * ramp_samples >= length:
            return envelope

        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples))
        envelope[:ramp_samples] = ramp
        envelope[-ramp_samples:] = ramp[::-1]
        return envelope

    @staticmethod
    def _rayleigh_envelope(length: int, rng: np.random.Generator) -> np.ndarray:
        """Slowly varying Rayleigh amplitude, normalised to unit mean power.

        Generated by filtering white noise rather than drawing independent samples, because
        multipath fading is correlated in time: it puts nulls of milliseconds into an over,
        not per-sample noise.
        """
        # A fade rate of a few tens of Hz corresponds to walking pace at UHF.
        control_points = max(4, length // 20_000)
        coarse = rng.standard_normal(control_points) + 1j * rng.standard_normal(control_points)
        fine = np.interp(
            np.linspace(0, control_points - 1, length), np.arange(control_points), coarse.real
        ) + 1j * np.interp(
            np.linspace(0, control_points - 1, length), np.arange(control_points), coarse.imag
        )
        envelope = np.abs(fine)
        return envelope / np.sqrt(np.mean(envelope**2))

    # ----------------------------------------------------------------------------------
    # Serialisation
    # ----------------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Scenario:
        """Build a scenario from a parsed YAML document."""
        emitters = []
        for entry in payload.get("emitters", []):
            transmissions = [
                Transmission(start_s=float(item["start_s"]), stop_s=float(item["stop_s"]))
                for item in entry.get("transmissions", [])
            ]
            emitters.append(
                Emitter(
                    **{k: v for k, v in entry.items() if k != "transmissions"},
                    transmissions=transmissions,
                )
            )

        propagation = Propagation(**payload.get("propagation", {}))
        known = {"emitters", "propagation"}
        return cls(
            **{k: v for k, v in payload.items() if k not in known},
            emitters=emitters,
            propagation=propagation,
        )

    @classmethod
    def load(cls, path: Path) -> Scenario:
        """Load a scenario from a YAML file.

        Args:
            path: Path to the scenario document.

        Returns:
            The parsed scenario.
        """
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    def to_dict(self) -> dict[str, Any]:
        """Return the scenario as a plain dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write the scenario to a YAML file.

        Args:
            path: Destination file.
        """
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
