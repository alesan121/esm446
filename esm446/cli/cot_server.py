"""Serve Cursor-on-Target over TLS for a TAK client to connect to.

The replacement for `legacy/cot_server.py`, which was a separate program the rest of the
system never fed. Here the server is a transport like any other: it can be run on its own to
receive events on standard input, or constructed by the node and published to with the same
call as UDP or TCP.

Running it on its own is what makes it testable by hand -- pipe a stored capture's events in
and watch them appear in iTAK -- without needing a receiver connected.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from esm446.io.cot_transport import DEFAULT_PORT, TlsServerTransport

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the TLS server from the command line.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cert", type=Path, required=True, help="server certificate (PEM)")
    parser.add_argument("--key", type=Path, required=True, help="server private key (PEM)")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to bind; pass 0.0.0.0 to accept clients from the network",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to bind")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="read CoT events from standard input and broadcast them, one per line",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
    logging.getLogger().setLevel(logging.INFO)

    try:
        server = TlsServerTransport(args.cert, args.key, args.host, args.port)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    try:
        if args.stdin:
            for line in sys.stdin:
                if line.strip():
                    server.publish([line.strip()])
        else:
            logger.info("cot: serving; connect a TAK client and Ctrl-C to stop")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("cot: stopping")
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
