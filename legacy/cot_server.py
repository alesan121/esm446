#!/usr/bin/env python3
"""
cot_server.py — Servidor TCP/TLS CoT para iTAK
iTAK conecta con SSL y recibe mensajes CoT en tiempo real.
"""

import socket
import ssl
import threading
import sys
from pathlib import Path

HOST     = "0.0.0.0"
PORT     = 4242
CERT     = Path(__file__).parent / "cot_server.crt"
KEY      = Path(__file__).parent / "cot_server.key"

_clients: list[socket.socket] = []
_lock    = threading.Lock()


def _build_ssl_context() -> ssl.SSLContext:
    """
    Contexto TLS servidor — cifrado obligatorio, sin verificar cliente.
    Analogía: el servidor tiene su tarjeta de identidad (cert),
    pero no exige que el cliente traiga la suya.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=CERT, keyfile=KEY)
    # iTAK iOS no envía certificado cliente en conexión básica
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _handle_client(conn: ssl.SSLSocket, addr: tuple) -> None:
    print(f"[CoT] iTAK conectado: {addr[0]}:{addr[1]}", flush=True)
    with _lock:
        _clients.append(conn)
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
    except (ConnectionResetError, OSError, ssl.SSLError):
        pass
    finally:
        with _lock:
            if conn in _clients:
                _clients.remove(conn)
        conn.close()
        print(f"[CoT] iTAK desconectado: {addr[0]}", flush=True)


def broadcast(cot_xml: str) -> None:
    """Envía CoT a todos los clientes TLS conectados."""
    payload = cot_xml.encode("utf-8")
    dead    = []
    with _lock:
        for client in _clients:
            try:
                client.sendall(payload)
            except OSError:
                dead.append(client)
        for d in dead:
            _clients.remove(d)


def serve_forever() -> None:
    if not CERT.exists() or not KEY.exists():
        print("[ERROR] Certificados no encontrados. Ejecuta primero:", flush=True)
        print("  openssl req -x509 -newkey rsa:2048 -nodes \\")
        print("    -keyout cot_server.key -out cot_server.crt \\")
        print("    -days 365 -subj '/CN=EW-Suite' \\")
        print("    -addext 'subjectAltName=IP:192.168.2.66'")
        sys.exit(1)

    ctx    = _build_ssl_context()
    raw    = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    raw.bind((HOST, PORT))
    raw.listen(5)
    server = ctx.wrap_socket(raw, server_side=True)

    print(f"[CoT] Servidor TLS activo en {HOST}:{PORT}", flush=True)
    print(f"[CoT] Cert: {CERT}", flush=True)
    print(f"[CoT] iTAK → SSL → 192.168.2.66:{PORT}", flush=True)

    while True:
        try:
            conn, addr = server.accept()
            t = threading.Thread(
                target=_handle_client, args=(conn, addr), daemon=True
            )
            t.start()
        except ssl.SSLError as e:
            print(f"[WARN] Handshake fallido: {e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        serve_forever()

    elif sys.argv[1] == "--inject":
        # Inyecta CoT desde stdin al servidor local (sin verificar cert)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        sock = socket.create_connection(("127.0.0.1", PORT))
        tls  = ctx.wrap_socket(sock)
        for line in sys.stdin:
            tls.sendall(line.encode())
        tls.close()
