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
from bob.research.s12 import (
    DEFAULT_TAU,
    STRATEGY,
    evaluate,
    print_escape_probability,
)
from helpers import RESEARCH_CLOSE, research_settled, seed_research_yes_quote

runner = CliRunner()


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def _seed_closes(connection, closes: dict[int, str]) -> None:
    bars = [
        MinuteBar(
            end_ts=checkpoint_end_ts(RESEARCH_CLOSE, minute),
            open=price,
            high=price,
            low=price,
            close=price,
        )
        for minute, price in sorted(closes.items())
    ]
    store_btc_candles(connection, bars)


def test_print_escape_probability_zero_hops() -> None:
    closes = [Decimal("150")] * 55
    escape = print_escape_probability(
        closes,
        checkpoint=55,
        floor=Decimal("100"),
        cap=Decimal("199.99"),
    )
    assert escape == Decimal("0")


def test_s12_trades_when_escape_below_threshold(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    _seed_closes(db, {minute: "150" for minute in range(1, 56)})

    report = evaluate(db, minutes=(55,), side="yes", tau=DEFAULT_TAU)

    assert report.strategy == STRATEGY == "s12"
    assert report.tau == DEFAULT_TAU
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1
    assert report.minutes[0].abstained == 0


def test_s12_abstains_when_print_risk_high(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    # Matched-horizon hops of ~80 fill half the escape mass on each side.
    closes = {
        minute: "110" if (minute // 5) % 2 == 0 else "190" for minute in range(1, 56)
    }
    _seed_closes(db, closes)

    report = evaluate(db, minutes=(55,), side="yes", tau=Decimal("0.75"))

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"print_risk_high": 1}


def test_s12_tau_is_configurable(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    closes = {minute: "150" for minute in range(1, 56)}
    for minute in range(1, 56):
        # Mild hops: some escape mass but not always.
        closes[minute] = "150" if minute % 2 == 0 else "170"
    _seed_closes(db, closes)

    strict = evaluate(db, minutes=(55,), side="yes", tau=Decimal("0.95"))
    loose = evaluate(db, minutes=(55,), side="yes", tau=Decimal("0.55"))

    assert strict.minutes[0].eligible <= loose.minutes[0].eligible


def test_research_s12_cli_tau(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    _seed_closes(connection, {minute: "150" for minute in range(1, 56)})
    seed_research_yes_quote(connection, ticker="KXBTC-99JUN0108-A", minute=55)
    connection.close()

    result = runner.invoke(
        app,
        [
            "research",
            "s12",
            "--db",
            str(db_path),
            "--minutes",
            "55",
            "--tau",
            "0.75",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "s12" in result.output
    assert "tau=0.75" in result.output
    assert "100.0%" in result.output
