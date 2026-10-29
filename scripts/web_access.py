"""Who may use the Web UI, and from where.

- API requests must come from the Web UI's own page. A request sent by another website's page
  (Origin or Sec-Fetch-Site says so) is refused, so a page open in the same browser cannot ask
  questions, start updates, or point the API settings at another server to collect the stored key.
  While the UI listens on this computer only, the Host header must name this computer as well, so a
  page cannot pass as it by pointing its own domain at 127.0.0.1 (DNS rebinding).
- Opened to the network (--host 0.0.0.0), the UI requires an access password (LKA_WEB_PASSWORD).
  Managing documents and settings stays limited to this computer; colleagues only ask questions.
"""

from __future__ import annotations

import hmac
import ipaddress
import secrets
import threading
import time
from http.cookies import SimpleCookie
from urllib.parse import urlparse

SESSION_COOKIE = "lka_session"
SESSION_SECONDS = 7 * 24 * 3600
LOGIN_FAILURES_ALLOWED = 10
LOGIN_LOCKOUT_SECONDS = 600

LOOPBACK_HOST_NAMES = {"127.0.0.1", "localhost", "::1"}


def is_loopback_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def is_loopback_bind(host: str) -> bool:
    return host.strip("[]").lower() in LOOPBACK_HOST_NAMES or is_loopback_address(host.strip("[]"))


def host_name(host_header: str) -> str:
    """«127.0.0.1:8765» -> «127.0.0.1»; «[::1]:8765» -> «::1»."""
    host = host_header.strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def cross_site_reason(headers, local_only: bool) -> str:
    """Why an API request did not come from the Web UI's own page; "" when it did (or from a script)."""
    host = str(headers.get("Host") or "")
    if local_only and host_name(host) not in LOOPBACK_HOST_NAMES:
        return "unexpected host"
    if str(headers.get("Sec-Fetch-Site") or "").lower() in {"cross-site", "same-site"}:
        return "cross-site request"
    origin = headers.get("Origin")
    if origin is not None:
        parsed = urlparse(str(origin))
        if str(origin) == "null" or parsed.netloc.lower() != host.lower():
            return "cross-origin request"
    return ""


class Access:
    """The access password and the signed-in browsers (kept in memory: a restart signs everyone out)."""

    def __init__(self, password: str) -> None:
        self.password = password
        self._sessions: dict[str, float] = {}
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @property
    def required(self) -> bool:
        return bool(self.password)

    def signed_in(self, cookie_header: str | None) -> bool:
        if not self.required:
            return True
        token = self._token(cookie_header)
        now = time.time()
        with self._lock:
            expires = self._sessions.get(token or "")
            if not expires or expires < now:
                self._sessions.pop(token or "", None)
                return False
        return True

    def sign_in(self, client: str, password: str) -> str | None:
        """A new session token for the right password; None otherwise or while the client is locked out."""
        now = time.time()
        with self._lock:
            recent = [moment for moment in self._failures.get(client, []) if now - moment < LOGIN_LOCKOUT_SECONDS]
            self._failures[client] = recent
            if len(recent) >= LOGIN_FAILURES_ALLOWED:
                return None
        if not hmac.compare_digest(password.encode("utf-8"), self.password.encode("utf-8")):
            with self._lock:
                self._failures.setdefault(client, []).append(now)
            time.sleep(1)
            return None
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._failures.pop(client, None)
            self._sessions[token] = now + SESSION_SECONDS
        return token

    def locked_out(self, client: str) -> bool:
        now = time.time()
        with self._lock:
            return len([m for m in self._failures.get(client, []) if now - m < LOGIN_LOCKOUT_SECONDS]) >= LOGIN_FAILURES_ALLOWED

    def sign_out(self, cookie_header: str | None) -> None:
        token = self._token(cookie_header)
        with self._lock:
            self._sessions.pop(token or "", None)

    @staticmethod
    def _token(cookie_header: str | None) -> str | None:
        if not cookie_header:
            return None
        cookie = SimpleCookie()
        try:
            cookie.load(cookie_header)
        except Exception:  # noqa: BLE001
            return None
        morsel = cookie.get(SESSION_COOKIE)
        return morsel.value if morsel else None


# Lax: opening the UI from a link keeps the session. Everything that changes something is a JSON
# POST, which Lax never sends from another site and the origin check refuses anyway.
def session_cookie(token: str) -> str:
    return f"{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={SESSION_SECONDS}"


def cleared_cookie() -> str:
    return f"{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"
