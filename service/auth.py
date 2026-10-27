"""HTTP Basic authentication for the demo deployment."""

from __future__ import annotations

import base64
import binascii
import secrets

USERNAME = "judge"


def check_basic_auth(header: str | None, password: str | None) -> bool:
    """Return True if ``header`` carries the expected credentials.

    With no password configured (development mode) every request is accepted.
    Comparison is constant-time.
    """
    if not password:
        return True
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    user, sep, given = decoded.partition(":")
    if not sep:
        return False
    user_ok = secrets.compare_digest(user.encode(), USERNAME.encode())
    pass_ok = secrets.compare_digest(given.encode(), password.encode())
    return user_ok and pass_ok
