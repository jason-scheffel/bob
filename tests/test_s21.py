# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.cli import app
from bob.db import connect, initialize_schema, store_btc_candles, store_settled_events
from bob.research.s21 import (
    STRATEGY,
    evaluate,
    residual_lock_z,
    rms_close_diff_sigma,
)
from helpers import research_flat_bars, research_settled

runner = CliRunner()


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_rms_and_lock_helpers() -> None:
    flat = [Decimal("150")] * 10
    assert rms_close_diff_sigma(flat) == 0
    assert residual_lock_z(
        edge_distance=Decimal("40"),
        sigma=Decimal("10"),
        remaining=16,
    ) == Decimal("1")


def test_s21_trades_when_path_is_flat_and_interior(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, research_flat_bars(range(1, 56), "150"))

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.strategy == STRATEGY == "s21"
    assert report.z_star == Decimal("0.50")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1


def test_s21_abstains_when_lock_is_weak(db) -> None:
    store_settled_events(db, [research_settled(expiration="105", winner="a")])
    # Large one-minute swings, then finish near the floor → small z.
    bars = research_flat_bars(range(1, 50), "150")
    bars.extend(research_flat_bars(range(50, 53), "180"))
    bars.extend(research_flat_bars(range(53, 56), "105"))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("weak_lock", 0) == 1


def test_research_s21_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    store_btc_candles(connection, research_flat_bars(range(1, 56), "150"))
    connection.close()

    result = runner.invoke(
        app,
        ["research", "s21", "--db", str(db_path), "--minutes", "55"],
    )

    assert result.exit_code == 0, result.output
    assert "s21" in result.output
    assert "residual-range lock" in result.output
