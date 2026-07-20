# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import timedelta
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
from bob.research.s13 import (
    DEFAULT_P_STAR,
    STRATEGY,
    evaluate,
    terminal_win_probability,
)
from helpers import RESEARCH_CLOSE, research_settled, seed_research_yes_quote

runner = CliRunner()


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def _seed_hour(
    connection,
    close_ts,
    *,
    price: str = "150",
    high: str | None = None,
    low: str | None = None,
) -> None:
    hi = high if high is not None else price
    lo = low if low is not None else price
    bars = [
        MinuteBar(
            end_ts=checkpoint_end_ts(close_ts, minute),
            open=price,
            high=hi,
            low=lo,
            close=price,
        )
        for minute in range(1, 60)
    ]
    store_btc_candles(connection, bars)


def test_terminal_win_probability_center_is_high() -> None:
    p = terminal_win_probability(
        price=Decimal("150"),
        floor=Decimal("100"),
        cap=Decimal("199.99"),
        sigma=Decimal("10"),
        remaining=10,
    )
    assert p > Decimal("0.85")


def test_s13_trades_when_quiet_and_centered(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    prior = RESEARCH_CLOSE - timedelta(hours=1)
    _seed_hour(db, prior, price="150", high="151", low="149")
    _seed_hour(db, RESEARCH_CLOSE, price="150", high="151", low="149")

    report = evaluate(db, minutes=(55,), side="yes", p_star=DEFAULT_P_STAR)

    assert report.strategy == STRATEGY == "s13"
    assert report.p_star == DEFAULT_P_STAR
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1


def test_s13_abstains_when_terminal_prob_low(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    prior = RESEARCH_CLOSE - timedelta(hours=1)
    _seed_hour(db, prior, price="150", high="250", low="50")
    _seed_hour(db, RESEARCH_CLOSE, price="150", high="250", low="50")

    report = evaluate(db, minutes=(55,), side="yes", p_star=Decimal("0.70"))

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"low_terminal_prob": 1}


def test_s13_p_star_is_configurable(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    prior = RESEARCH_CLOSE - timedelta(hours=1)
    _seed_hour(db, prior, price="150", high="170", low="130")
    _seed_hour(db, RESEARCH_CLOSE, price="150", high="170", low="130")

    strict = evaluate(db, minutes=(55,), side="yes", p_star=Decimal("0.95"))
    loose = evaluate(db, minutes=(55,), side="yes", p_star=Decimal("0.55"))

    assert strict.minutes[0].eligible <= loose.minutes[0].eligible


def test_research_s13_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    prior = RESEARCH_CLOSE - timedelta(hours=1)
    _seed_hour(connection, prior, price="150", high="151", low="149")
    _seed_hour(connection, RESEARCH_CLOSE, price="150", high="151", low="149")
    seed_research_yes_quote(connection, ticker="KXBTC-99JUN0108-A", minute=55)
    connection.close()

    result = runner.invoke(
        app,
        [
            "research",
            "s13",
            "--db",
            str(db_path),
            "--minutes",
            "55",
            "--p-star",
            "0.70",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "s13" in result.output
    assert "p_star=0.70" in result.output
    assert "100.0%" in result.output
