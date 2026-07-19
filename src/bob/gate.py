# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys
from collections.abc import Callable
from getpass import getpass

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

DISCLAIMER = """\
Do not use this shit. I am not responsible for anything or any losses.

It currently does not work as expected.

This software is experimental and also not finished and may fail and lose funds.
It is public for my own convenience and not because I care about sharing it with people.

This is just a hobby trading bot thats fun for me. But I vibe code this repo too.
I think that you might not be right in the head if you actually use this, so dont.\
"""

# Argon2id PHC string (params + salt + digest). No plaintext password in-repo.
PASSWORD_HASH = (
    "$argon2id$v=19$m=19456,t=2,p=1$"
    "ehg58+Q2STs/MpNWHlP9og$"
    "bO47ZbhjdeePGkxCM/NGhzCM+vdjYcWwqkMLEGKEEo4"
)

_HASHER = PasswordHasher(memory_cost=19456, time_cost=2, parallelism=1)


def verify_password(candidate: str, password_hash: str = PASSWORD_HASH) -> bool:
    try:
        return _HASHER.verify(password_hash, candidate)
    except VerifyMismatchError:
        return False


def require_gate(
    *,
    prompt: Callable[[str], str] = getpass,
    verifier: Callable[[str], bool] | None = None,
) -> None:
    print(DISCLAIMER, flush=True)
    print(flush=True)
    check = verifier if verifier is not None else verify_password
    try:
        candidate = prompt("Password: ")
    except (EOFError, KeyboardInterrupt):
        _deny()
    if not candidate or not check(candidate):
        _deny()


def _deny() -> None:
    print("Access denied.", file=sys.stderr)
    raise SystemExit(1)
