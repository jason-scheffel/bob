# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.cli import app
from bob.db import (
    MinuteBar,
    connect,
    initialize_schema,
    store_btc_candles,
    store_settled_events,
)
from bob.research.common import checkpoint_end_ts
from bob.research.s14 import (
    DEFAULT_P_STAR,
    DEFAULT_Q_STAR,
    STRATEGY,
    evaluate,
)
from helpers import RESEARCH_CLOSE, research_settled

runner = CliRunner()


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def _seed_storm_then_calm(connection, close_ts, *, calm_from: int) -> None:
    """Wild early bars, quiet trailing bars; close pinned at 150."""
    bars = []
    for minute in range(1, 60):
        if minute < calm_from:
            high, low = "250", "50"
        else:
            high, low = "151", "149"
        bars.append(
            MinuteBar(
                end_ts=checkpoint_end_ts(close_ts, minute),
                open="150",
                high=high,
                low=low,
                close="150",
            )
        )
    store_btc_candles(connection, bars)


def _seed_uniform(connection, close_ts, *, high: str, low: str) -> None:
    bars = [
        MinuteBar(
            end_ts=checkpoint_end_ts(close_ts, minute),
            open="150",
            high=high,
            low=low,
            close="150",
        )
        for minute in range(1, 60)
    ]
    store_btc_candles(connection, bars)


def test_s14_trades_on_storm_then_calm(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    # M45 → R=15 → calm minutes 31..45
    _seed_storm_then_calm(db, RESEARCH_CLOSE, calm_from=31)

    report = evaluate(
        db,
        minutes=(45,),
        side="yes",
        p_star=DEFAULT_P_STAR,
        q_star=DEFAULT_Q_STAR,
    )

    assert report.strategy == STRATEGY == "s14"
    assert report.p_star == DEFAULT_P_STAR
    assert report.q_star == DEFAULT_Q_STAR
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1


def test_s14_abstains_already_priced_when_whole_hour_quiet(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    _seed_uniform(db, RESEARCH_CLOSE, high="151", low="149")

    report = evaluate(db, minutes=(45,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("already_priced", 0) == 1


def test_s14_abstains_low_quality_when_recent_still_wild(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    _seed_uniform(db, RESEARCH_CLOSE, high="250", low="50")

    report = evaluate(db, minutes=(45,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("low_quality", 0) == 1


def test_s14_thresholds_are_configurable(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    _seed_storm_then_calm(db, RESEARCH_CLOSE, calm_from=31)

    strict = evaluate(
        db, minutes=(45,), p_star=Decimal("0.95"), q_star=Decimal("0.40")
    )
    loose = evaluate(
        db, minutes=(45,), p_star=Decimal("0.55"), q_star=Decimal("0.80")
    )

    assert strict.minutes[0].eligible <= loose.minutes[0].eligible


def test_research_s14_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    _seed_storm_then_calm(connection, RESEARCH_CLOSE, calm_from=31)
    connection.close()

    result = runner.invoke(
        app,
        [
            "research",
            "s14",
            "--db",
            str(db_path),
            "--minutes",
            "45",
            "--p-star",
            "0.70",
            "--q-star",
            "0.60",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "s14" in result.output
    assert "p_star=0.70" in result.output
    assert "q_star=0.60" in result.output
