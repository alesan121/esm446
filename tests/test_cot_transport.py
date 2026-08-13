"""Verification of the CoT transports.

Two things matter here and they pull in opposite directions. The message must not depend on
the wire it leaves over -- that is the whole point of having one abstraction instead of v0's
two unrelated paths -- and a wire that fails must not take the capture with it.

Everything runs against loopback sockets in-process. No network, no fixtures outside the test.
"""

from __future__ import annotations

import socket
import ssl
import subprocess
import threading
import time
from pathlib import Path

import pytest

from esm446.core.node import EmissionReport
from esm446.io.cot import ReceiverSite, events_for
from esm446.io.cot_transport import (
    CotSink,
    MultiTransport,
    NullTransport,
    TcpTransport,
    TlsServerTransport,
    TlsTransport,
    UdpTransport,
    open_transport,
)

BASE_TIME = 1_786_950_000.0
SITE = ReceiverSite(latitude=40.4168, longitude=-3.7038)


def emission(estimated_dbm: float | None = None, calibrated: bool = False) -> EmissionReport:
    return EmissionReport(
        timestamp=BASE_TIME,
        frequency_hz=446_093_757.0,
        pmr_channel=8,
        bin_index=9,
        duration_s=4.2,
        peak_power_dbfs=-0.6,
        snr_db=40.9,
        estimated_dbm=estimated_dbm,
        calibrated=calibrated,
        ctcss_tone_hz=114.8,
        classification="FRIEND",
        offset_s=0.0,
        peak_deviation_hz=1_347.0,
        gains={},
    )


@pytest.fixture
def certificate(tmp_path: Path) -> tuple[Path, Path]:
    """A throwaway self-signed pair, generated per test."""
    if not shutil_which("openssl"):
        pytest.skip("openssl is not installed")
    cert, key = tmp_path / "cot.crt", tmp_path / "cot.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        capture_output=True,
        check=True,
    )
    return cert, key


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


class Collector:
    """A loopback server that records what it was sent."""

    def __init__(self, kind: str, certificate: tuple[Path, Path] | None = None) -> None:
        self.received: list[bytes] = []
        self._kind = kind
        self._certificate = certificate
        if kind == "udp":
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.bind(("127.0.0.1", 0))
        else:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(("127.0.0.1", 0))
            self._socket.listen(1)
        self._socket.settimeout(5.0)
        self.host, self.port = self._socket.getsockname()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            if self._kind == "udp":
                while True:
                    data, _ = self._socket.recvfrom(65535)
                    self.received.append(data)
            else:
                connection, _ = self._socket.accept()
                if self._kind == "tls":
                    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    context.load_cert_chain(str(self._certificate[0]), str(self._certificate[1]))
                    connection = context.wrap_socket(connection, server_side=True)
                with connection:
                    while chunk := connection.recv(65535):
                        self.received.append(chunk)
        except (OSError, ssl.SSLError, TimeoutError):
            pass

    def wait_for(self, count: int, timeout_s: float = 3.0) -> None:
        """Block until at least ``count`` bytes have arrived, or give up."""
        deadline = time.monotonic() + timeout_s
        while sum(len(c) for c in self.received) < count and time.monotonic() < deadline:
            time.sleep(0.02)

    def wait_for_events(self, count: int, timeout_s: float = 3.0) -> None:
        """Block until ``count`` events have arrived.

        Counting bytes is not enough: a byte of the first datagram satisfies a byte count
        while the second is still in flight, which is exactly how this raced before.
        """
        deadline = time.monotonic() + timeout_s
        while self.payload().count(b"<event") < count and time.monotonic() < deadline:
            time.sleep(0.02)

    def payload(self) -> bytes:
        return b"".join(self.received)

    def close(self) -> None:
        self._socket.close()


# --------------------------------------------------------------------------------------
# The property the whole abstraction exists for
# --------------------------------------------------------------------------------------


def test_the_same_emission_is_identical_over_every_transport(certificate) -> None:
    """v0 had two publishing paths that produced different messages. This is why there is one.

    The event is built once and handed to the transport as bytes, so a byte comparison across
    the three wires is the strongest available statement that the wire cannot change what a
    consumer sees.
    """
    events = events_for(emission(), SITE)
    expected = "".join(events).encode("utf-8")

    udp = Collector("udp")
    tcp = Collector("tcp")
    tls = Collector("tls", certificate)
    try:
        with UdpTransport(udp.host, udp.port) as transport:
            transport.publish(events)
        with TcpTransport(tcp.host, tcp.port) as transport:
            transport.publish(events)
        with TlsTransport(tls.host, tls.port, verify=False) as transport:
            transport.publish(events)

        for collector in (udp, tcp, tls):
            collector.wait_for(len(expected))
            assert collector.payload() == expected, f"{collector._kind} altered the message"
    finally:
        for collector in (udp, tcp, tls):
            collector.close()


# --------------------------------------------------------------------------------------
# Individual transports
# --------------------------------------------------------------------------------------


def test_udp_publishes_one_datagram_per_event() -> None:
    collector = Collector("udp")
    try:
        events = events_for(emission(), SITE)
        with UdpTransport(collector.host, collector.port) as transport:
            assert transport.publish(events) == len(events)
        collector.wait_for_events(len(events))
        assert len(collector.received) == len(events), "one event per datagram"
    finally:
        collector.close()


def test_tcp_reconnects_rather_than_giving_up() -> None:
    """A TAK server restarts. The node must recover without being restarted itself."""
    collector = Collector("tcp")
    transport = TcpTransport(collector.host, collector.port)
    try:
        assert transport.publish(["<event/>"]) == 1
        transport.close()  # as if the link dropped
        assert transport.publish(["<event/>"]) == 1, "did not reconnect"
    finally:
        transport.close()
        collector.close()


def test_a_dead_link_does_not_raise() -> None:
    """The rule the sinks follow: losing the feed must never cost the capture."""
    with TcpTransport("127.0.0.1", 1) as transport:
        assert transport.publish(["<event/>"]) == 0


def test_a_dead_link_backs_off_instead_of_hammering() -> None:
    """Retrying every emission against a server that is down helps nobody."""
    transport = TcpTransport("127.0.0.1", 1)
    started = time.monotonic()
    for _ in range(5):
        transport.publish(["<event/>"])
    elapsed = time.monotonic() - started
    transport.close()

    assert elapsed < 2.0, "each attempt paid a full connection timeout"


def test_tls_verifies_certificates_unless_told_not_to(certificate) -> None:
    """A feed that silently accepts any certificate is not protected against anybody."""
    collector = Collector("tls", certificate)
    try:
        with TlsTransport(collector.host, collector.port, timeout_s=1.0) as transport:
            assert transport.publish(["<event/>"]) == 0, "an unknown CA was accepted"
    finally:
        collector.close()


# --------------------------------------------------------------------------------------
# The server iTAK connects to
# --------------------------------------------------------------------------------------


def test_the_tls_server_broadcasts_to_a_connected_client(certificate) -> None:
    """The replacement for legacy/cot_server.py, reached through the same publish call."""
    cert, key = certificate
    server = TlsServerTransport(cert, key, host="127.0.0.1", port=0)
    try:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        with context.wrap_socket(socket.create_connection((server.host, server.port))) as client:
            deadline = time.monotonic() + 3.0
            while server.client_count == 0 and time.monotonic() < deadline:
                time.sleep(0.02)

            events = events_for(emission(), SITE)
            assert server.publish(events) == len(events)

            client.settimeout(3.0)
            received = client.recv(65535)

        assert received.decode("utf-8") == "".join(events)
    finally:
        server.close()


def test_the_server_publishes_nothing_with_nobody_listening(certificate) -> None:
    """Not an error. A sensor with no client attached still has to keep capturing."""
    cert, key = certificate
    with TlsServerTransport(cert, key, host="127.0.0.1", port=0) as server:
        assert server.publish(events_for(emission(), SITE)) == 0


def test_the_server_says_what_to_run_when_the_certificate_is_missing(tmp_path: Path) -> None:
    """The v0 server printed the openssl line too. It is the one genuinely useful part."""
    with pytest.raises(FileNotFoundError, match="openssl req"):
        TlsServerTransport(tmp_path / "absent.crt", tmp_path / "absent.key")


# --------------------------------------------------------------------------------------
# Composition and configuration
# --------------------------------------------------------------------------------------


def test_a_broken_transport_does_not_stop_the_others() -> None:
    collector = Collector("udp")
    try:

        class Broken(NullTransport):
            def publish(self, events: list[str]) -> int:
                raise OSError("interface down")

        good = UdpTransport(collector.host, collector.port)
        with MultiTransport([Broken(), good]) as transport:
            assert transport.publish(["<event/>"]) == 1
        collector.wait_for(1)
        assert collector.payload() == b"<event/>"
    finally:
        collector.close()


def test_open_transport_selects_by_scheme() -> None:
    assert isinstance(open_transport("udp://127.0.0.1:4242"), UdpTransport)
    assert isinstance(open_transport("tcp://127.0.0.1:4242"), TcpTransport)
    assert isinstance(open_transport("tls://127.0.0.1:4242"), TlsTransport)


def test_no_destination_publishes_nowhere() -> None:
    """The default. An unconfigured node must not put traffic on a network."""
    transport = open_transport(None)

    assert isinstance(transport, NullTransport)
    assert transport.publish(["<event/>"]) == 0


def test_an_unknown_scheme_is_refused() -> None:
    with pytest.raises(ValueError, match="udp, tcp or tls"):
        open_transport("smtp://127.0.0.1:25")


def test_a_destination_without_a_host_is_refused() -> None:
    with pytest.raises(ValueError, match="no host"):
        open_transport("udp://")


# --------------------------------------------------------------------------------------
# The sink the node actually writes to
# --------------------------------------------------------------------------------------


def test_the_sink_publishes_a_track_per_emission() -> None:
    collector = Collector("udp")
    try:
        with CotSink(UdpTransport(collector.host, collector.port), SITE) as sink:
            assert sink.write([emission(), emission()]) == 2
        collector.wait_for_events(2)
        assert collector.payload().count(b"<event") == 2
    finally:
        collector.close()


def test_the_sink_sends_no_rings_for_an_uncalibrated_emission() -> None:
    """Every emission this system currently produces. One track, no rings, ce unknown."""
    collector = Collector("udp")
    try:
        with CotSink(UdpTransport(collector.host, collector.port), SITE) as sink:
            sink.write([emission()])
        collector.wait_for_events(1)
        payload = collector.payload()

        assert payload.count(b"<event") == 1
        assert b'ce="9999999.0"' in payload
        assert b"u-d-r" not in payload
    finally:
        collector.close()


def test_the_sink_writing_nothing_is_not_an_error() -> None:
    with CotSink(NullTransport()) as sink:
        assert sink.write([]) == 0
