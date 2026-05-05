import json
import socket
import time

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def emit(port: int, event_type: str, payload: dict) -> None:
    msg = {"type": event_type, "timestamp": time.time(), **payload}
    _sock.sendto(json.dumps(msg).encode(), ('127.0.0.1', port))