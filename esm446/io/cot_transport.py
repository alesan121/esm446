"""One way out for Cursor-on-Target, whichever wire it happens to be on.

v0 had two paths that never spoke to each other. `legacy/EW_FoF_Scanner.sh` pushed datagrams
with ``nc -u`` while accumulating XML in a shell variable inside a pipeline subshell -- so the
buffer could not outlive the loop that filled it -- and `legacy/cot_server.py` was a separate
TLS server for iTAK to connect to that the rest of the system never called. Two ways to
publish, neither aware of the other, and a message went out over one or the other depending
on which script happened to run.

Here there is one interface. The node calls `CotTransport.publish` and does not know whether
the bytes leave over UDP, TCP, TLS, or to a client that connected to it. The event itself is
built once by `esm446.io.cot`, so the same emission produces the same XML on every transport
-- which is a property worth testing rather than assuming, and `tests/test_cot_transport.py`
does.

Which transport is which
------------------------
**UDP** is fire and forget. Nothing tells you the message arrived and nothing retries, which
is the right trade for a stream of position updates where the next one is along shortly.

**TCP** gets delivery and ordering, and pays for them by blocking when the far end stops
reading. Reconnection is on a backoff, because a TAK server that has just restarted will
refuse connections for a while and hammering it helps nobody.

**TLS** is TCP with the transport encrypted. TAK deployments generally require it.

**A TLS server** is the shape iTAK actually wants: the client connects to the sensor rather
than the other way round. This replaces `legacy/cot_server.py` and, unlike it, is fed by the
same `publish` call as everything else.

Failure policy
--------------
The same rule the sinks follow, for the same reason: **losing the feed must never cost the
capture**. Every send is guarded, a failure is logged, and the node keeps processing. A
transport that cannot deliver drops the message rather than buffering without limit, because
a queue that grows while a link is down eventually takes the process with it.
"""

from __future__ import annotations

import logging
import socket
import ssl
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

from esm446.core.geolocation import estimate_from_report
from esm446.io.cot import events_for
from esm446.io.sinks import EmissionSink

logger = logging.getLogger(__name__)

#: Default port for a TAK feed.
DEFAULT_PORT = 4242

#: Seconds to wait for a connection or a send before giving up on it.
DEFAULT_TIMEOUT_S = 5.0

#: Backoff bounds for reconnecting a stream transport, in seconds. A server that has just
#: restarted refuses connections for a while, and retrying every frame helps nobody.
_BACKOFF_START_S = 1.0
_BACKOFF_MAX_S = 30.0


class CotTransport(ABC):
    """Somewhere CoT messages go."""

    @abstractmethod
    def publish(self, events: list[str]) -> int:
        """Send events.

        Args:
            events: CoT XML documents.

        Returns:
            Number of events handed to the wire. Zero when the link is down, which is not
            an error: the capture matters more than the feed.
        """

    def close(self) -> None:
        """Release any resource held. Safe to call more than once."""

    def __enter__(self) -> CotTransport:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class UdpTransport(CotTransport):
    """Datagrams to a host and port, or to a multicast group.

    Nothing confirms arrival and nothing is retried. For a feed of tracks that are refreshed
    as emissions complete, a lost message costs one update rather than the connection.
    """

    def __init__(self, host: str, port: int = DEFAULT_PORT, ttl: int = 1) -> None:
        self.host = host
        self.port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        logger.info("cot: publishing over UDP to %s:%d", host, port)

    def publish(self, events: list[str]) -> int:
        sent = 0
        for event in events:
            try:
                self._socket.sendto(event.encode("utf-8"), (self.host, self.port))
                sent += 1
            except OSError:
                logger.warning("cot: UDP send to %s:%d failed", self.host, self.port)
        return sent

    def close(self) -> None:
        self._socket.close()


class StreamTransport(CotTransport):
    """A connected client transport, with reconnection. Base of the TCP and TLS clients."""

    def __init__(self, host: str, port: int = DEFAULT_PORT, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._socket: socket.socket | None = None
        self._next_attempt = 0.0
        self._backoff_s = _BACKOFF_START_S

    def _wrap(self, raw: socket.socket) -> socket.socket:
        """Hook for a subclass to put a layer on top of the connected socket."""
        return raw

    def _connect(self) -> socket.socket | None:
        """Connect, or decline to try again yet.

        Returns:
            The connected socket, or ``None`` while the backoff is still running.
        """
        if self._socket is not None:
            return self._socket
        if time.monotonic() < self._next_attempt:
            return None
        try:
            raw = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            self._socket = self._wrap(raw)
            self._backoff_s = _BACKOFF_START_S
            logger.info("cot: connected to %s:%d", self.host, self.port)
        except (OSError, ssl.SSLError) as error:
            self._next_attempt = time.monotonic() + self._backoff_s
            logger.warning(
                "cot: cannot reach %s:%d (%s); retrying in %.0f s",
                self.host,
                self.port,
                error,
                self._backoff_s,
            )
            self._backoff_s = min(self._backoff_s * 2, _BACKOFF_MAX_S)
        return self._socket

    def publish(self, events: list[str]) -> int:
        connection = self._connect()
        if connection is None:
            return 0
        sent = 0
        for event in events:
            try:
                connection.sendall(event.encode("utf-8"))
                sent += 1
            except (OSError, ssl.SSLError):
                logger.warning(
                    "cot: send to %s:%d failed, dropping the connection", self.host, self.port
                )
                self.close()
                break
        return sent

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None


class TcpTransport(StreamTransport):
    """A plain TCP client. Delivery and ordering, at the cost of blocking on a stalled peer."""

    def __init__(self, host: str, port: int = DEFAULT_PORT, timeout_s: float = DEFAULT_TIMEOUT_S):
        super().__init__(host, port, timeout_s)
        logger.info("cot: publishing over TCP to %s:%d", host, port)


class TlsTransport(StreamTransport):
    """A TLS client, which is what most TAK servers require.

    Certificate verification is on by default and switching it off is an explicit argument,
    not a default. A feed that silently accepts any certificate is not encrypted against
    anybody who can get in the path, which is the threat encryption is there for.
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        ca_file: Path | None = None,
        client_cert: Path | None = None,
        client_key: Path | None = None,
        verify: bool = True,
    ) -> None:
        super().__init__(host, port, timeout_s)
        self._context = ssl.create_default_context(cafile=str(ca_file) if ca_file else None)
        if not verify:
            # Self-signed certificates are the norm on a private TAK deployment, so this has
            # to be reachable -- but as a decision somebody made, recorded in the log.
            self._context.check_hostname = False
            self._context.verify_mode = ssl.CERT_NONE
            logger.warning(
                "cot: TLS certificate verification is disabled for %s:%d; the feed is "
                "encrypted but the far end is not authenticated",
                host,
                port,
            )
        if client_cert is not None:
            self._context.load_cert_chain(str(client_cert), str(client_key) if client_key else None)
        logger.info("cot: publishing over TLS to %s:%d", host, port)

    def _wrap(self, raw: socket.socket) -> socket.socket:
        return self._context.wrap_socket(raw, server_hostname=self.host)


class TlsServerTransport(CotTransport):
    """A TLS server that fans messages out to whatever connected to it.

    The shape iTAK wants: the client dials the sensor. This is `legacy/cot_server.py` brought
    inside the interface, so the node publishes to it with the same call it uses for every
    other transport instead of the server being a separate program nothing fed.

    Accepting runs on a background thread. Broadcast walks the client list under a lock and
    drops anything that fails, since a client that has gone away must not stall the capture.
    """

    def __init__(
        self,
        certfile: Path,
        keyfile: Path,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
    ) -> None:
        """Bind and start accepting.

        The default binds to loopback only. A sensor feed reachable from every interface the
        moment somebody starts it is an exposure nobody asked for; putting it on the network
        is a decision, made by passing an address.
        """
        if not Path(certfile).exists() or not Path(keyfile).exists():
            raise FileNotFoundError(
                f"no certificate at {certfile} and key at {keyfile}. Generate a self-signed "
                f"pair with:\n"
                f"  openssl req -x509 -newkey rsa:2048 -nodes -days 365 "
                f"-keyout {keyfile} -out {certfile} -subj '/CN=ESM-446'"
            )
        self._context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._context.load_cert_chain(certfile=str(certfile), keyfile=str(keyfile))
        # A TAK client on a private deployment does not present a certificate of its own.
        self._context.verify_mode = ssl.CERT_NONE

        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(5)
        self._listener.settimeout(0.5)

        self.host, self.port = self._listener.getsockname()
        self._clients: list[ssl.SSLSocket] = []
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._accept_forever, daemon=True)
        self._thread.start()
        logger.info("cot: TLS server listening on %s:%d", self.host, self.port)

    def _accept_forever(self) -> None:
        """Accept connections until closed. A failed handshake costs that client only."""
        while self._running:
            try:
                raw, address = self._listener.accept()
            except (TimeoutError, OSError):
                continue
            try:
                connection = self._context.wrap_socket(raw, server_side=True)
            except (ssl.SSLError, OSError) as error:
                logger.warning("cot: handshake from %s failed: %s", address[0], error)
                raw.close()
                continue
            with self._lock:
                self._clients.append(connection)
            logger.info("cot: client connected from %s:%d", address[0], address[1])

    @property
    def client_count(self) -> int:
        """How many clients are currently connected."""
        with self._lock:
            return len(self._clients)

    def publish(self, events: list[str]) -> int:
        payload = "".join(events).encode("utf-8")
        if not payload:
            return 0
        dead: list[ssl.SSLSocket] = []
        with self._lock:
            for client in self._clients:
                try:
                    client.sendall(payload)
                except (OSError, ssl.SSLError):
                    dead.append(client)
            for client in dead:
                self._clients.remove(client)
                client.close()
        if dead:
            logger.info("cot: dropped %d client(s) that stopped reading", len(dead))
        # Reported as events published, not as events times clients: the caller is asking
        # whether the message left, not how many screens it reached.
        return len(events) if self.client_count else 0

    def close(self) -> None:
        self._running = False
        with self._lock:
            for client in self._clients:
                client.close()
            self._clients.clear()
        self._listener.close()
        self._thread.join(timeout=2.0)


class MultiTransport(CotTransport):
    """Publish to several transports at once, surviving any of them failing."""

    def __init__(self, transports: list[CotTransport]) -> None:
        self.transports = transports

    def publish(self, events: list[str]) -> int:
        published = 0
        for transport in self.transports:
            try:
                published = max(published, transport.publish(events))
            except Exception:  # noqa: BLE001 -- a broken feed must not stop the capture
                logger.exception("cot: %s failed to publish", type(transport).__name__)
        return published

    def close(self) -> None:
        for transport in self.transports:
            try:
                transport.close()
            except Exception:  # noqa: BLE001
                logger.exception("cot: %s failed to close", type(transport).__name__)


class NullTransport(CotTransport):
    """Publishes nowhere. The default, so an unconfigured node emits nothing to a network."""

    def publish(self, events: list[str]) -> int:
        return 0


class CotSink(EmissionSink):
    """Publish emissions to a TAK feed as they complete.

    An `esm446.io.sinks.EmissionSink`, so the node writes to it exactly as it writes to the
    archive and does not know a network is involved. That is what puts the feed on the same
    footing as the store: one call site, one failure policy, and the two cannot drift apart.

    Ranges come from `esm446.core.geolocation`, which returns ``None`` for an uncalibrated
    report -- so the track goes out with an unknown circular error rather than a ring nobody
    measured.
    """

    def __init__(
        self,
        transport: CotTransport,
        site: Any | None = None,
        stale_s: float = 300.0,
        with_range: bool = True,
    ) -> None:
        self.transport = transport
        self.site = site
        self.stale_s = stale_s
        self.with_range = with_range
        self.published = 0

    def write(self, reports: list[Any]) -> int:
        published = 0
        for report in reports:
            estimate = estimate_from_report(report) if self.with_range else None
            events = events_for(report, self.site, estimate, self.stale_s)
            published += self.transport.publish(events)
        self.published += published
        return published

    def close(self) -> None:
        self.transport.close()


def open_transport(destination: str | None) -> CotTransport:
    """Build a transport from a URL.

    ``udp://host:port``, ``tcp://host:port``, ``tls://host:port``, or ``None`` for no feed.
    Selecting the transport by configuration is the point of the abstraction: the node is
    written once and the deployment decides the wire.

    Args:
        destination: The URL, or ``None``.

    Returns:
        The transport. `NullTransport` when nothing was configured.

    Raises:
        ValueError: If the scheme is not one of the three, or the URL has no host.
    """
    if not destination:
        return NullTransport()

    parsed = urlparse(destination)
    if not parsed.hostname:
        raise ValueError(f"no host in CoT destination {destination!r}")
    port = parsed.port or DEFAULT_PORT

    if parsed.scheme == "udp":
        return UdpTransport(parsed.hostname, port)
    if parsed.scheme == "tcp":
        return TcpTransport(parsed.hostname, port)
    if parsed.scheme == "tls":
        return TlsTransport(parsed.hostname, port)
    raise ValueError(f"unknown CoT transport {parsed.scheme!r}; use udp, tcp or tls")
