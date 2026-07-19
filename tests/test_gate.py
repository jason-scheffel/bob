# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pytest
from argon2 import PasswordHasher

from bob.gate import DISCLAIMER, PASSWORD_HASH, require_gate, verify_password

_FIXTURE_PASSWORD = "correct horse battery"
_HASHER = PasswordHasher(memory_cost=19456, time_cost=2, parallelism=1)
_FIXTURE_HASH = _HASHER.hash(_FIXTURE_PASSWORD)


def test_disclaimer_matches_readme() -> None:
    readme = Path(__file__).resolve().parents[1].joinpath("README.md").read_text()
    lines: list[str] = []
    in_alert = False
    for line in readme.splitlines():
        if line.startswith("> [!IMPORTANT]"):
            in_alert = True
            continue
        if in_alert:
            if line.startswith("> "):
                lines.append(line[2:])
            elif line.startswith(">"):
                lines.append(line[1:])
            else:
                break
    assert "\n".join(lines) == DISCLAIMER


def test_password_hash_params() -> None:
    assert PASSWORD_HASH.startswith("$argon2id$v=19$m=19456,t=2,p=1$")


def test_verify_password_accepts_fixture() -> None:
    assert verify_password(_FIXTURE_PASSWORD, _FIXTURE_HASH) is True


def test_verify_password_rejects_wrong() -> None:
    assert verify_password("wrong password", _FIXTURE_HASH) is False


def test_verify_password_rejects_empty() -> None:
    assert verify_password("", _FIXTURE_HASH) is False


def test_require_gate_success(capsys: pytest.CaptureFixture[str]) -> None:
    require_gate(prompt=lambda _: "any", verifier=lambda _: True)
    assert DISCLAIMER in capsys.readouterr().out


def test_require_gate_disclaimer_before_prompt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def prompt(_msg: str) -> str:
        assert DISCLAIMER in capsys.readouterr().out
        return "x"

    require_gate(prompt=prompt, verifier=lambda _: True)


def test_require_gate_mismatch_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        require_gate(prompt=lambda _: "nope", verifier=lambda _: False)
    assert exc.value.code == 1
    assert "Access denied." in capsys.readouterr().err


def test_require_gate_empty_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        require_gate(prompt=lambda _: "", verifier=lambda _: True)
    assert exc.value.code == 1
    assert "Access denied." in capsys.readouterr().err


def test_require_gate_eof_exits(capsys: pytest.CaptureFixture[str]) -> None:
    def prompt(_msg: str) -> str:
        raise EOFError

    with pytest.raises(SystemExit) as exc:
        require_gate(prompt=prompt, verifier=lambda _: True)
    assert exc.value.code == 1
    assert "Access denied." in capsys.readouterr().err


def test_require_gate_keyboard_interrupt_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def prompt(_msg: str) -> str:
        raise KeyboardInterrupt

    with pytest.raises(SystemExit) as exc:
        require_gate(prompt=prompt, verifier=lambda _: True)
    assert exc.value.code == 1
    assert "Access denied." in capsys.readouterr().err
