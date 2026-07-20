# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.cli import app
from bob.db import connect, initialize_schema, store_btc_candles, store_settled_events
from bob.research.s22 import (
    STRATEGY,
    asymmetric_buffers_clear,
    evaluate,
    wall_distances,
)
from helpers import research_flat_bars, research_settled

runner = CliRunner()


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_wall_distances_and_buffers() -> None:
    assert wall_distances(Decimal("130"), "100", "199.99") == (
        Decimal("30"),
        Decimal("69.99"),
    )
    assert asymmetric_buffers_clear(
        d_near=Decimal("30"),
        d_far=Decimal("70"),
        sigma=Decimal("10"),
        remaining=16,
        near_mult=Decimal("0.25"),
        far_mult=Decimal("0.75"),
    )
    assert not asymmetric_buffers_clear(
        d_near=Decimal("5"),
        d_far=Decimal("95"),
        sigma=Decimal("10"),
        remaining=16,
        near_mult=Decimal("0.25"),
        far_mult=Decimal("0.75"),
    )


def test_s22_trades_when_flat_and_interior(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, research_flat_bars(range(1, 56), "150"))

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.strategy == STRATEGY == "s22"
    assert report.near_mult == Decimal("0.25")
    assert report.far_mult == Decimal("0.75")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1


def test_s22_abstains_when_near_wall_is_too_thin(db) -> None:
    store_settled_events(db, [research_settled(expiration="105", winner="a")])
    bars = research_flat_bars(range(1, 50), "150")
    bars.extend(research_flat_bars(range(50, 53), "180"))
    bars.extend(research_flat_bars(range(53, 56), "105"))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("asymmetric_weak", 0) == 1


def test_research_s22_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    store_btc_candles(connection, research_flat_bars(range(1, 56), "150"))
    connection.close()

    result = runner.invoke(
        app,
        ["research", "s22", "--db", str(db_path), "--minutes", "55"],
    )

    assert result.exit_code == 0, result.output
    assert "s22" in result.output
    assert "asymmetric dual-buffer" in result.output
