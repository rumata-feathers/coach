"""Rate-limiter singleton shared by main.py and routes.py.

Keyed on X-Forwarded-For (first hop) so Railway's reverse proxy does not
make every request look like it originates from the same IP.
Falls back to request.client.host for local / direct connections.
"""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _client_ip(request: Request) -> str:
    """Return the real client IP, honouring Railway's proxy headers."""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_client_ip)
