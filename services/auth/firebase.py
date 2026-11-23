"""Firebase ID-token verification.

Verifies Firebase Authentication ID tokens (RS256 JWTs) using Google's public
signing certificates. This only needs the Firebase **project id** — no service
account or admin credentials are required for verification.

The token's signature, expiry, audience (project id), and issuer are all
checked, so a token minted for another project cannot be replayed here.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

import httpx
import jwt

from config import settings

_CERTS_URL = (
    "https://www.googleapis.com/robot/v1/metadata/x509/"
    "securetoken@system.gserviceaccount.com"
)
_DEFAULT_TTL_SECONDS = 3600

_lock = threading.Lock()
_certs: dict[str, str] = {}
_certs_expire_at: float = 0.0


class FirebaseAuthError(Exception):
    """Raised when an ID token cannot be verified."""


def _cache_ttl(headers: httpx.Headers) -> int:
    match = re.search(r"max-age=(\d+)", headers.get("cache-control", ""))
    return int(match.group(1)) if match else _DEFAULT_TTL_SECONDS


def _load_certs(force: bool = False) -> dict[str, str]:
    global _certs, _certs_expire_at
    now = time.time()
    if not force and _certs and _certs_expire_at > now:
        return _certs
    with _lock:
        if not force and _certs and _certs_expire_at > now:
            return _certs
        try:
            resp = httpx.get(_CERTS_URL, timeout=10)
            resp.raise_for_status()
            _certs = dict(resp.json())
            _certs_expire_at = time.time() + _cache_ttl(resp.headers)
        except (httpx.HTTPError, ValueError) as exc:
            if _certs:
                # Serve stale certs rather than failing every request during a blip.
                return _certs
            raise FirebaseAuthError("Unable to fetch Firebase signing keys") from exc
    return _certs


def verify_firebase_id_token(token: str) -> dict[str, Any]:
    """Verify a Firebase ID token and return its claims.

    Raises :class:`FirebaseAuthError` on any problem.
    """
    project_id = (settings.firebase_project_id or "").strip()
    if not project_id:
        raise FirebaseAuthError("FIREBASE_PROJECT_ID is not configured")

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise FirebaseAuthError("Malformed token header") from exc

    kid = header.get("kid")
    if not kid:
        raise FirebaseAuthError("Token is missing a key id")

    certs = _load_certs()
    pem = certs.get(kid)
    if pem is None:
        pem = _load_certs(force=True).get(kid)
    if pem is None:
        raise FirebaseAuthError("Unknown signing key")

    try:
        claims = jwt.decode(
            token,
            pem,
            algorithms=["RS256"],
            audience=project_id,
            issuer=f"https://securetoken.google.com/{project_id}",
            options={"require": ["exp", "iat", "aud", "iss", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise FirebaseAuthError("Invalid or expired token") from exc

    if not claims.get("sub"):
        raise FirebaseAuthError("Token is missing a subject")
    return claims
