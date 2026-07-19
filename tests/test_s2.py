# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.browse import BracketRow
from bob.cli import app
from bob.db import (
    MinuteBar,
    connect,
    initialize_schema,
    store_btc_candles,
    store_settled_events,
)
from bob.kalshi import STATUS_COMPLETE, Bracket, Event, SettledEvent
from bob.research.common import checkpoint_end_ts
from bob.research.s2 import (
    STRATEGY,
    in_center_band,
    lookback_minutes,
    path_stays_in_bracket,
    evaluate,
)

runner = CliRunner()
CLOSE = datetime(2099, 6, 1, 12, 0, tzinfo=timezone.utc)


def _settled(
    *,
    expiration: str = "150",
    winner: str = "mid",
) -> SettledEvent:
    event_ticker = "KXBTC-99JUN0108"
    return SettledEvent(
        event=Event(
            event_ticker=event_ticker,
            close_ts=CLOSE,
            status=STATUS_COMPLETE,
            expiration_value=Decimal(expiration),
        ),
        brackets=(
            Bracket(
                ticker=f"{event_ticker}-LOW",
                event_ticker=event_ticker,
                floor_strike=None,
                cap_strike=Decimal("99.99"),
                won=winner == "low",
            ),
            Bracket(
                ticker=f"{event_ticker}-MID",
                event_ticker=event_ticker,
                floor_strike=Decimal("100"),
                cap_strike=Decimal("199.99"),
                won=winner == "mid",
            ),
            Bracket(
                ticker=f"{event_ticker}-HIGH",
                event_ticker=event_ticker,
                floor_strike=Decimal("200"),
                cap_strike=None,
                won=winner == "high",
            ),
        ),
    )


def _bar(end_ts: int, *, low: str, high: str, close: str) -> MinuteBar:
    return MinuteBar(
        end_ts=end_ts,
        open=close,
        high=high,
        low=low,
        close=close,
    )


def _seed_stable_center(connection, *, close: str = "150") -> None:
    """Five bars 51..55 all inside mid bracket, close centered."""
    bars = []
    for minute in range(51, 56):
        bars.append(
            _bar(
                checkpoint_end_ts(CLOSE, minute),
                low="140",
                high="160",
                close=close if minute == 55 else "150",
            )
        )
    store_btc_candles(connection, bars)


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_lookback_minutes() -> None:
    assert lookback_minutes(55) == (51, 52, 53, 54, 55)
    with pytest.raises(ValueError, match=">="):
        lookback_minutes(4)


def test_in_center_band() -> None:
    # width 100: middle 50% is [125, 175]
    assert in_center_band(Decimal("150"), "100", "199.99")
    assert not in_center_band(Decimal("110"), "100", "199.99")
    assert in_center_band(Decimal("150"), "100", "200") is True
    assert in_center_band(Decimal("150"), None, "200") is None


def test_path_stays_in_bracket() -> None:
    bracket = BracketRow("m", "100", "199.99", True)
    assert path_stays_in_bracket(
        [(Decimal("160"), Decimal("140"))],
        bracket,
    )
    assert not path_stays_in_bracket(
        [(Decimal("210"), Decimal("140"))],
        bracket,
    )


def test_evaluate_takes_stable_center_trade(db) -> None:
    store_settled_events(db, [_settled(expiration="150", winner="mid")])
    _seed_stable_center(db)
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.strategy == STRATEGY == "s2"
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1
    assert report.minutes[0].abstained == 0


def test_evaluate_abstains_near_edge(db) -> None:
    store_settled_events(db, [_settled(expiration="150", winner="mid")])
    bars = []
    for minute in range(51, 56):
        bars.append(
            _bar(
                checkpoint_end_ts(CLOSE, minute),
                low="105",
                high="120",
                close="110",
            )
        )
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"near_edge": 1}


def test_evaluate_abstains_unstable_path(db) -> None:
    store_settled_events(db, [_settled(expiration="150", winner="mid")])
    bars = []
    for minute in range(51, 56):
        if minute == 53:
            bars.append(
                _bar(
                    checkpoint_end_ts(CLOSE, minute),
                    low="140",
                    high="210",
                    close="150",
                )
            )
        else:
            bars.append(
                _bar(
                    checkpoint_end_ts(CLOSE, minute),
                    low="140",
                    high="160",
                    close="150",
                )
            )
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].abstentions == {"unstable_path": 1}


def test_evaluate_abstains_open_ended(db) -> None:
    store_settled_events(db, [_settled(expiration="250", winner="high")])
    bars = []
    for minute in range(51, 56):
        bars.append(
            _bar(
                checkpoint_end_ts(CLOSE, minute),
                low="220",
                high="260",
                close="250",
            )
        )
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].abstentions == {"open_ended_bracket": 1}


def test_research_s2_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [_settled()])
    _seed_stable_center(connection)
    connection.close()

    result = runner.invoke(
        app,
        ["research", "s2", "--db", str(db_path), "--minutes", "55"],
    )
    assert result.exit_code == 0, result.output
    assert "s2" in result.output
    assert "100.0%" in result.output
    assert "abstained" in result.output
