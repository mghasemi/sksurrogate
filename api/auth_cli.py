"""Command-line account management for the SKSurrogate control plane.

The API home is shared with the server (``SKSURROGATE_API_HOME``), so accounts
created here are immediately usable by a running instance that points at the
same directory.

Usage::

    python -m api.auth_cli create-admin --username mehdi            # prompts
    python -m api.auth_cli add-user --username analyst --password-stdin
    python -m api.auth_cli list-users
    python -m api.auth_cli set-password --username mehdi
    python -m api.auth_cli disable --username analyst
    python -m api.auth_cli create-key --username mehdi --name ci

Passwords are read from a prompt or ``--password-stdin`` by default so they
never land in shell history. ``--password`` exists for container entrypoints but
is discouraged.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from . import auth
from .config import settings


def _read_password(args) -> str:
    if args.password:
        return args.password
    if args.password_stdin:
        value = sys.stdin.readline().rstrip("\n")
        if not value:
            raise SystemExit("empty password on stdin")
        return value
    first = getpass.getpass("Password: ")
    second = getpass.getpass("Repeat password: ")
    if first != second:
        raise SystemExit("passwords do not match")
    return first


def _store():
    settings.ensure()
    return auth.store()


def cmd_create_admin(args) -> int:
    store = _store()
    if store.count_users() > 0:
        print(
            "Refusing: the user table is not empty. Use `add-user --admin` instead.",
            file=sys.stderr,
        )
        return 2
    user = store.create_user(args.username, _read_password(args), is_admin=True)
    print("Created administrator %r (id %d)" % (user["username"], user["user_id"]))
    return 0


def cmd_add_user(args) -> int:
    user = _store().create_user(args.username, _read_password(args), is_admin=args.admin)
    print(
        "Created %s %r (id %d)"
        % ("administrator" if user["is_admin"] else "user", user["username"], user["user_id"])
    )
    return 0


def cmd_list_users(args) -> int:
    users = _store().list_users()
    if not users:
        print("No accounts. Create one with `create-admin --username <name>`.")
        return 0
    print("%-4s %-24s %-6s %-7s %s" % ("id", "username", "admin", "active", "created"))
    for user in users:
        print(
            "%-4d %-24s %-6s %-7s %s"
            % (
                user["user_id"],
                user["username"],
                "yes" if user["is_admin"] else "no",
                "yes" if user["is_active"] else "no",
                user["created_at"],
            )
        )
    return 0


def cmd_set_password(args) -> int:
    store = _store()
    user = store.get_user_by_username(args.username)
    if user is None:
        print("No such user: %r" % args.username, file=sys.stderr)
        return 2
    store.set_password(user["user_id"], _read_password(args))
    print("Password updated for %r; their sessions were revoked." % user["username"])
    return 0


def cmd_disable(args) -> int:
    store = _store()
    user = store.get_user_by_username(args.username)
    if user is None:
        print("No such user: %r" % args.username, file=sys.stderr)
        return 2
    store.set_active(user["user_id"], False)
    print("Disabled %r and revoked their sessions." % user["username"])
    return 0


def cmd_enable(args) -> int:
    store = _store()
    user = store.get_user_by_username(args.username)
    if user is None:
        print("No such user: %r" % args.username, file=sys.stderr)
        return 2
    store.set_active(user["user_id"], True)
    print("Enabled %r." % user["username"])
    return 0


def cmd_create_key(args) -> int:
    store = _store()
    user = store.get_user_by_username(args.username)
    if user is None:
        print("No such user: %r" % args.username, file=sys.stderr)
        return 2
    secret, _record = store.create_api_key(user["user_id"], args.name)
    print("API key for %r (shown once — store it now):" % user["username"])
    print(secret)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m api.auth_cli",
        description="Manage SKSurrogate control-plane accounts and API keys.",
    )
    parser.add_argument(
        "--api-home",
        default=None,
        help="storage root (defaults to SKSURROGATE_API_HOME or ./var/sksurrogate-api)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_credentials(command: argparse.ArgumentParser) -> None:
        command.add_argument("--password", default=None, help="literal password (discouraged)")
        command.add_argument(
            "--password-stdin", action="store_true", help="read the password from stdin"
        )

    create = sub.add_parser("create-admin", help="create the first administrator")
    create.add_argument("--username", required=True)
    add_credentials(create)
    create.set_defaults(func=cmd_create_admin)

    add = sub.add_parser("add-user", help="create another account")
    add.add_argument("--username", required=True)
    add.add_argument("--admin", action="store_true")
    add_credentials(add)
    add.set_defaults(func=cmd_add_user)

    listing = sub.add_parser("list-users", help="list accounts")
    listing.set_defaults(func=cmd_list_users)

    password = sub.add_parser("set-password", help="reset an account password")
    password.add_argument("--username", required=True)
    add_credentials(password)
    password.set_defaults(func=cmd_set_password)

    disable = sub.add_parser("disable", help="deactivate an account")
    disable.add_argument("--username", required=True)
    disable.set_defaults(func=cmd_disable)

    enable = sub.add_parser("enable", help="reactivate an account")
    enable.add_argument("--username", required=True)
    enable.set_defaults(func=cmd_enable)

    key = sub.add_parser("create-key", help="mint an API key for an account")
    key.add_argument("--username", required=True)
    key.add_argument("--name", default="cli")
    key.set_defaults(func=cmd_create_key)

    args = parser.parse_args(argv)

    if args.api_home:
        import pathlib

        os.environ["SKSURROGATE_API_HOME"] = args.api_home
        settings.root = pathlib.Path(args.api_home).resolve()
        auth.reset_store()

    try:
        return args.func(args)
    except auth.AuthError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
