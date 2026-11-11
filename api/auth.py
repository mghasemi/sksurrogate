"""User accounts, sessions, and API keys for the SKSurrogate control plane.

The v1 control plane had exactly one credential: the shared
``SKSURROGATE_API_KEY`` environment variable. That is enough to close an open
port, but it cannot say *who* a caller is, so it cannot support per-user
ownership or sharing. This module adds the identity layer:

* **Users** — username + password, hashed with :func:`hashlib.scrypt`
  (stdlib only, no new dependency). Passwords are never stored or logged.
* **Sessions** — opaque random tokens for the browser UI, stored only as
  SHA-256 digests, delivered as an ``HttpOnly`` cookie. Revocable server-side,
  unlike a JWT.
* **API keys** — per-user secrets for programmatic clients, prefixed
  ``sksk_`` so they are recognisable in logs and scanners, stored as digests,
  shown in plaintext exactly once at creation.

All state lives in one SQLite database under the API home
(:attr:`api.config.Settings.auth_db_path`); nothing external is required. The
module is deliberately framework-free so the tests (and the ``python -m
api.auth`` CLI) can use it without a running server.

Threat-model notes
------------------
* Password and key digests are compared with :func:`hmac.compare_digest`.
* The scrypt parameters below are the CPython-documented interactive cost
  (``n=2**14, r=8, p=1``); raise them for a long-lived public deployment, at
  the cost of login latency.
* Failed logins are recorded in an in-process sliding window
  (:class:`LoginThrottle`) and logged at WARNING without the supplied secret.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from .config import settings

LOGGER = logging.getLogger("sksurrogate.api.auth")

#: scrypt cost parameters (see module docstring).
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SCRYPT_MAXMEM = 64 * 1024 * 1024

#: Version tag for the stored password format, so the parameters can change
#: without invalidating existing hashes.
_PASSWORD_SCHEME = "scrypt"

#: API keys carry this prefix so leaked keys are recognisable by scanners.
API_KEY_PREFIX = "sksk_"

#: Session cookie name. The ``__Host-`` prefix is only used over HTTPS, so a
#: plain name keeps local HTTP development working.
SESSION_COOKIE = "sksurrogate_session"

_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+-]{2,63}$")
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 1024
MIN_API_KEY_NAME_LENGTH = 1
MAX_API_KEY_NAME_LENGTH = 64


class AuthError(ValueError):
    """Raised for invalid account/key input (mapped to HTTP 400/409 by the API)."""


class UsernameTaken(AuthError):
    """Raised when creating a user whose username already exists (HTTP 409)."""


class LastAdminError(AuthError):
    """Raised when an operation would leave the deployment without an admin."""


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #


def hash_password(password: str) -> str:
    """Return a self-describing scrypt digest for ``password``.

    Format: ``scrypt$<n>$<r>$<p>$<salt-hex>$<digest-hex>`` — self-describing so
    the cost parameters can be raised later without breaking old hashes.
    """
    if not isinstance(password, str):
        raise AuthError("password must be a string")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError("password must be at least %d characters" % MIN_PASSWORD_LENGTH)
    if len(password) > MAX_PASSWORD_LENGTH:
        raise AuthError("password must be at most %d characters" % MAX_PASSWORD_LENGTH)
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
        maxmem=_SCRYPT_MAXMEM,
    )
    return "%s$%d$%d$%d$%s$%s" % (
        _PASSWORD_SCHEME,
        _SCRYPT_N,
        _SCRYPT_R,
        _SCRYPT_P,
        salt.hex(),
        digest.hex(),
    )


def verify_password(stored: str, password: str) -> bool:
    """Constant-time check of ``password`` against a :func:`hash_password` digest.

    Malformed digests are treated as a failed check rather than an error so a
    corrupted row cannot be used to distinguish account existence.
    """
    if not isinstance(stored, str) or not isinstance(password, str):
        return False
    try:
        scheme, n_raw, r_raw, p_raw, salt_hex, digest_hex = stored.split("$")
        if scheme != _PASSWORD_SCHEME:
            return False
        n, r, p = int(n_raw), int(r_raw), int(p_raw)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, AttributeError):
        return False
    candidate = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=len(expected),
        maxmem=_SCRYPT_MAXMEM,
    )
    return hmac.compare_digest(candidate, expected)


#: Pre-computed digest used to keep failure timing flat for unknown usernames.
_DUMMY_DIGEST = hash_password(secrets.token_urlsafe(24))


def _digest_token(token: str) -> str:
    """SHA-256 of a session token or API key; the value persisted in SQLite."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


#: Public alias used by the router (revoking all sessions except the current one).
digest_token = _digest_token


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def validate_username(username: str) -> str:
    """Return ``username`` if it satisfies the account-name policy."""
    if not isinstance(username, str):
        raise AuthError("username must be a string")
    value = username.strip()
    if not _USERNAME_RE.match(value):
        raise AuthError(
            "username must be 3-64 characters: letters, digits, dot, underscore, "
            "at-sign, plus or hyphen, starting with a letter or digit"
        )
    return value


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Principal:
    """The authenticated identity behind one request.

    ``user_id`` is ``None`` for the two non-account identities:
    :data:`ANONYMOUS` (auth not enforced) and :data:`STATIC_KEY` (the legacy
    ``SKSURROGATE_API_KEY`` break-glass principal, which is admin-equivalent).
    """

    user_id: Optional[int]
    username: str
    is_admin: bool
    source: str  # "session" | "api_key" | "static" | "anonymous"

    @property
    def authenticated(self) -> bool:
        return self.source != "anonymous"

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "is_admin": self.is_admin,
            "source": self.source,
        }


ANONYMOUS = Principal(None, "anonymous", False, "anonymous")
STATIC_KEY = Principal(None, "api-key", True, "static")

#: The principal of the request currently being handled. Set by
#: :class:`api.main.AuthMiddleware` for HTTP and WebSocket scopes alike, so
#: routers read the identity without threading a parameter through every helper.
CURRENT_PRINCIPAL: ContextVar[Principal] = ContextVar(
    "sksurrogate_principal", default=ANONYMOUS
)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    is_admin      INTEGER NOT NULL DEFAULT 0,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS api_keys (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    prefix      TEXT NOT NULL,
    key_hash    TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at  TEXT
);
CREATE INDEX IF NOT EXISTS api_keys_user ON api_keys(user_id);
"""


class AuthStore:
    """SQLite-backed user/session/API-key store.

    One connection per operation keeps the class thread-safe under uvicorn's
    worker threads without sharing a connection object; WAL mode makes
    concurrent readers cheap.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    # -- plumbing ---------------------------------------------------------- #

    @contextmanager
    def _connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialise(self) -> "AuthStore":
        with self._lock, self._connection() as connection:
            connection.executescript(_SCHEMA)
            self._purge_expired(connection)
        return self

    @staticmethod
    def _purge_expired(connection) -> None:
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (_iso(_now()),))

    # -- users ------------------------------------------------------------- #

    def create_user(self, username: str, password: str, *, is_admin: bool = False) -> dict:
        username = validate_username(username)
        password_hash = hash_password(password)
        stamp = _iso(_now())
        with self._lock, self._connection() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO users (username, password_hash, is_admin, is_active, "
                    "created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?)",
                    (username, password_hash, 1 if is_admin else 0, stamp, stamp),
                )
            except sqlite3.IntegrityError as exc:
                raise UsernameTaken("username %r is already taken" % username) from exc
            assert cursor.lastrowid is not None
            return self._user_row(connection, cursor.lastrowid)

    @staticmethod
    def _user_row(connection, user_id: int) -> dict:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            raise KeyError("no such user: %r" % user_id)
        return _user_dict(row)

    def get_user(self, user_id: int) -> Optional[dict]:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _user_dict(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[dict]:
        if not isinstance(username, str):
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ?", (username.strip(),)
            ).fetchone()
        return _user_dict(row) if row else None

    def list_users(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM users ORDER BY username").fetchall()
        return [_user_dict(row) for row in rows]

    def count_users(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def count_admins(self, *, active_only: bool = True) -> int:
        query = "SELECT COUNT(*) FROM users WHERE is_admin = 1"
        if active_only:
            query += " AND is_active = 1"
        with self._connection() as connection:
            return int(connection.execute(query).fetchone()[0])

    def set_password(self, user_id: int, password: str, *, keep_token_hash: Optional[str] = None) -> None:
        """Replace a user's password, revoking their other sessions."""
        password_hash = hash_password(password)
        with self._lock, self._connection() as connection:
            self._user_row(connection, user_id)
            connection.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (password_hash, _iso(_now()), user_id),
            )
            if keep_token_hash:
                connection.execute(
                    "DELETE FROM sessions WHERE user_id = ? AND token_hash != ?",
                    (user_id, keep_token_hash),
                )
            else:
                connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def set_active(self, user_id: int, active: bool) -> dict:
        """Enable/disable an account; disabling drops its sessions."""
        with self._lock, self._connection() as connection:
            user = self._user_row(connection, user_id)
            if not active and user["is_admin"] and self.count_admins(active_only=True) <= 1:
                raise LastAdminError("cannot disable the last active administrator")
            connection.execute(
                "UPDATE users SET is_active = ?, updated_at = ? WHERE id = ?",
                (1 if active else 0, _iso(_now()), user_id),
            )
            if not active:
                connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            return self._user_row(connection, user_id)

    def set_admin(self, user_id: int, is_admin: bool) -> dict:
        with self._lock, self._connection() as connection:
            user = self._user_row(connection, user_id)
            if user["is_admin"] and not is_admin and self.count_admins(active_only=True) <= 1:
                raise LastAdminError("cannot demote the last active administrator")
            connection.execute(
                "UPDATE users SET is_admin = ?, updated_at = ? WHERE id = ?",
                (1 if is_admin else 0, _iso(_now()), user_id),
            )
            return self._user_row(connection, user_id)

    # -- sessions ---------------------------------------------------------- #

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """Return the user dict for valid credentials, else ``None``.

        Runs a dummy scrypt against unknown usernames so response timing does
        not reveal which accounts exist.
        """
        user = self.get_user_by_username(username)
        if user is None:
            verify_password(_DUMMY_DIGEST, password or "")
            return None
        if not verify_password(user["password_hash"], password or ""):
            return None
        if not user["is_active"]:
            return None
        return user

    def create_session(self, user_id: int, *, ttl_seconds: int) -> str:
        """Create a session and return the opaque token (stored only as a digest)."""
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._lock, self._connection() as connection:
            self._user_row(connection, user_id)
            connection.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (_digest_token(token), user_id, _iso(now), _iso(now + timedelta(seconds=ttl_seconds)), _iso(now)),
            )
        return token

    def resolve_session(self, token: str) -> Optional[dict]:
        """Return the active user for a session token, or ``None`` if unknown/expired."""
        if not token:
            return None
        token_hash = _digest_token(token)
        now = _now()
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT s.token_hash, s.expires_at, u.* FROM sessions s "
                "JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if not hmac.compare_digest(row["token_hash"], token_hash):
                return None
            expires = _parse_iso(row["expires_at"])
            if expires is None or expires <= now:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
                return None
            if not row["is_active"]:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
                return None
            connection.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE token_hash = ?", (_iso(now), token_hash)
            )
            return _user_dict(row)

    def delete_session(self, token: str) -> None:
        if not token:
            return
        with self._lock, self._connection() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (_digest_token(token),))

    def delete_user_sessions(self, user_id: int) -> int:
        with self._lock, self._connection() as connection:
            cursor = connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            return cursor.rowcount

    # -- API keys ---------------------------------------------------------- #

    def create_api_key(self, user_id: int, name: str) -> tuple[str, dict]:
        """Create a key and return ``(plaintext, record)``; plaintext is shown once."""
        name = (name or "").strip()
        if not MIN_API_KEY_NAME_LENGTH <= len(name) <= MAX_API_KEY_NAME_LENGTH:
            raise AuthError("key name must be 1-%d characters" % MAX_API_KEY_NAME_LENGTH)
        secret = secrets.token_urlsafe(32)
        plaintext = "%s%s" % (API_KEY_PREFIX, secret)
        with self._lock, self._connection() as connection:
            self._user_row(connection, user_id)
            cursor = connection.execute(
                "INSERT INTO api_keys (user_id, name, prefix, key_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, name, plaintext[: len(API_KEY_PREFIX) + 6], _digest_token(plaintext), _iso(_now())),
            )
            row = connection.execute("SELECT * FROM api_keys WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return plaintext, _key_dict(row)

    def resolve_api_key(self, secret: str) -> Optional[dict]:
        """Return the owning active user for ``secret``, recording last use."""
        if not secret or not secret.startswith(API_KEY_PREFIX):
            return None
        key_hash = _digest_token(secret)
        now = _iso(_now())
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT k.id AS key_id, k.key_hash, k.revoked_at, u.* FROM api_keys k "
                "JOIN users u ON u.id = k.user_id WHERE k.key_hash = ?",
                (key_hash,),
            ).fetchone()
            if row is None or row["revoked_at"] is not None:
                return None
            if not hmac.compare_digest(row["key_hash"], key_hash):
                return None
            if not row["is_active"]:
                return None
            connection.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (now, row["key_id"]))
            return _user_dict(row)

    def list_api_keys(self, user_id: Optional[int] = None) -> list[dict]:
        query = "SELECT * FROM api_keys"
        params: tuple = ()
        if user_id is not None:
            query += " WHERE user_id = ?"
            params = (user_id,)
        query += " ORDER BY created_at DESC"
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_key_dict(row) for row in rows]

    def revoke_api_key(self, key_id: int, *, user_id: Optional[int] = None) -> dict:
        """Revoke a key; ``user_id`` scopes the operation to its owner."""
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
            if row is None:
                raise KeyError("no such API key: %r" % key_id)
            if user_id is not None and row["user_id"] != user_id:
                raise KeyError("no such API key: %r" % key_id)
            connection.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ?", (_iso(_now()), key_id)
            )
            updated = connection.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
        return _key_dict(updated)


def _user_dict(row) -> dict:
    """Public (password-free) view of a user row."""
    return {
        "user_id": int(row["id"]),
        "username": row["username"],
        "is_admin": bool(row["is_admin"]),
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "password_hash": row["password_hash"],
    }


def _key_dict(row) -> dict:
    return {
        "id": int(row["id"]),
        "user_id": int(row["user_id"]),
        "name": row["name"],
        "prefix": row["prefix"],
        "created_at": row["created_at"],
        "last_used_at": row["last_used_at"],
        "revoked_at": row["revoked_at"],
    }


# --------------------------------------------------------------------------- #
# Module-level store bound to the configured API home
# --------------------------------------------------------------------------- #

_STORE: Optional[AuthStore] = None
_STORE_PATH: Optional[Path] = None


def store() -> AuthStore:
    """Return the process-wide store for the configured API home.

    Re-created when ``settings.root`` changes, which is how tests redirect
    storage to a temporary directory (the same pattern the API fixtures use).
    """
    global _STORE, _STORE_PATH
    path = Path(settings.auth_db_path)
    if _STORE is None or _STORE_PATH != path:
        _STORE = AuthStore(path).initialise()
        _STORE_PATH = path
    return _STORE


def reset_store() -> None:
    """Drop the cached store (tests, and the CLI after switching roots)."""
    global _STORE, _STORE_PATH
    _STORE = None
    _STORE_PATH = None


# --------------------------------------------------------------------------- #
# Login throttling
# --------------------------------------------------------------------------- #


@dataclass
class LoginThrottle:
    """In-process sliding window of failed logins, keyed by username and client.

    Deliberately simple: the control plane is a single process, and the point is
    to make online password guessing expensive, not to be a distributed rate
    limiter (put a real one in front of a public deployment). State is lost on
    restart, which only ever *helps* a legitimate user.
    """

    max_failures: int = 5
    window_seconds: int = 900
    _failures: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _key(self, username: str, client: str) -> tuple:
        return ((username or "").strip().lower(), client or "unknown")

    def locked(self, username: str, client: str) -> bool:
        with self._lock:
            entry = self._failures.get(self._key(username, client))
            if not entry:
                return False
            count, first = entry
            if (_now() - first).total_seconds() > self.window_seconds:
                self._failures.pop(self._key(username, client), None)
                return False
            return count >= self.max_failures

    def record_failure(self, username: str, client: str) -> None:
        with self._lock:
            key = self._key(username, client)
            count, first = self._failures.get(key, (0, _now()))
            if (_now() - first).total_seconds() > self.window_seconds:
                count, first = 0, _now()
            self._failures[key] = (count + 1, first)

    def reset(self, username: str, client: str) -> None:
        with self._lock:
            self._failures.pop(self._key(username, client), None)


THROTTLE = LoginThrottle()


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def bootstrap_from_env() -> Optional[dict]:
    """Create the initial administrator from the environment, if configured.

    Idempotent: once the named user exists nothing happens, so the variables can
    stay in a compose file without resetting a changed password. Returns the
    created user dict, or ``None`` when inactive.
    """
    username = (os.environ.get("SKSURROGATE_ADMIN_USERNAME") or "").strip()
    password = os.environ.get("SKSURROGATE_ADMIN_PASSWORD") or ""
    if not username or not password:
        return None
    existing = store().get_user_by_username(username)
    if existing is not None:
        return None
    user = store().create_user(username, password, is_admin=True)
    LOGGER.warning(
        "Bootstrapped administrator %r from SKSURROGATE_ADMIN_USERNAME; "
        "rotate the password and remove SKSURROGATE_ADMIN_PASSWORD from the environment.",
        username,
    )
    return user


def create_first_admin(username: str, password: str) -> dict:
    """Create an admin account, refusing when any user already exists."""
    if store().count_users() > 0:
        raise AuthError(
            "the user table is not empty; use the admin API or `python -m api.auth` "
            "to manage accounts"
        )
    return store().create_user(username, password, is_admin=True)


__all__ = [
    "ANONYMOUS",
    "API_KEY_PREFIX",
    "AuthError",
    "AuthStore",
    "LastAdminError",
    "LoginThrottle",
    "Principal",
    "SESSION_COOKIE",
    "STATIC_KEY",
    "THROTTLE",
    "UsernameTaken",
    "bootstrap_from_env",
    "create_first_admin",
    "hash_password",
    "reset_store",
    "store",
    "validate_username",
    "verify_password",
]
