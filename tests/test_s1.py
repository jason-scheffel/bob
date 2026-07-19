# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from bob.browse import BracketRow, load_events
from bob.cli import app, parse_checkpoint_minutes
from bob.db import (
    MinuteBar,
    connect,
    initialize_schema,
    store_btc_candles,
    store_settled_events,
)
from bob.kalshi import STATUS_COMPLETE, Bracket, Event, SettledEvent
from bob.research.common import (
    bracket_containing,
    brackets_containing,
    checkpoint_end_ts,
    price_in_bracket,
)
from bob.research.s1 import STRATEGY, evaluate

runner = CliRunner()
CLOSE = datetime(2099, 6, 1, 12, 0, tzinfo=timezone.utc)


def _brackets(
    *,
    event_ticker: str = "KXBTC-99JUN0108",
    winner: str = "mid",
) -> tuple[Bracket, ...]:
    # Non-overlapping endpoints so inclusive mapping is unambiguous.
    items = (
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
    )
    return items


def _settled(
    *,
    event_ticker: str = "KXBTC-99JUN0108",
    close_ts: datetime = CLOSE,
    expiration: str = "150",
    winner: str = "mid",
) -> SettledEvent:
    return SettledEvent(
        event=Event(
            event_ticker=event_ticker,
            close_ts=close_ts,
            status=STATUS_COMPLETE,
            expiration_value=Decimal(expiration),
        ),
        brackets=_brackets(event_ticker=event_ticker, winner=winner),
    )


def _bar(end_ts: int, close: str) -> MinuteBar:
    return MinuteBar(
        end_ts=end_ts,
        open=close,
        high=close,
        low=close,
        close=close,
    )


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_price_in_bracket_boundaries() -> None:
    assert price_in_bracket(Decimal("100"), "100", "199.99")
    assert price_in_bracket(Decimal("199.99"), "100", "199.99")
    assert not price_in_bracket(Decimal("99.99"), "100", "199.99")
    assert not price_in_bracket(Decimal("200"), "100", "199.99")


def test_price_in_bracket_open_ends() -> None:
    assert price_in_bracket(Decimal("100"), None, "100")
    assert not price_in_bracket(Decimal("100.01"), None, "100")
    assert price_in_bracket(Decimal("200"), "200", None)
    assert not price_in_bracket(Decimal("199.99"), "200", None)
    assert not price_in_bracket(Decimal("150"), None, None)


def test_bracket_containing_exact_one() -> None:
    brackets = (
        BracketRow("a", None, "99.99", False),
        BracketRow("b", "100", "199.99", True),
        BracketRow("c", "200", None, False),
    )
    assert bracket_containing(Decimal("150"), brackets) == brackets[1]
    assert bracket_containing(Decimal("100"), brackets) == brackets[1]
    assert bracket_containing(Decimal("99.99"), brackets) == brackets[0]
    assert bracket_containing(Decimal("250"), brackets) == brackets[2]


def test_evaluate_ambiguous_shared_endpoint(db) -> None:
    event_ticker = "KXBTC-99JUN0108"
    store_settled_events(
        db,
        [
            SettledEvent(
                event=Event(
                    event_ticker=event_ticker,
                    close_ts=CLOSE,
                    status=STATUS_COMPLETE,
                    expiration_value=Decimal("150"),
                ),
                brackets=(
                    Bracket(
                        ticker=f"{event_ticker}-LOW",
                        event_ticker=event_ticker,
                        floor_strike=None,
                        cap_strike=Decimal("100"),
                        won=False,
                    ),
                    Bracket(
                        ticker=f"{event_ticker}-MID",
                        event_ticker=event_ticker,
                        floor_strike=Decimal("100"),
                        cap_strike=Decimal("199.99"),
                        won=True,
                    ),
                ),
            )
        ],
    )
    store_btc_candles(db, [_bar(checkpoint_end_ts(CLOSE, 45), "100")])
    report = evaluate(db, minutes=(45,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].exclusions == {"ambiguous_bracket": 1}


def test_bracket_containing_ambiguous_overlap() -> None:
    brackets = (
        BracketRow("a", "100", "200", False),
        BracketRow("b", "150", "250", True),
    )
    assert bracket_containing(Decimal("175"), brackets) is None
    assert len(brackets_containing(Decimal("175"), brackets)) == 2


def test_checkpoint_end_ts() -> None:
    assert checkpoint_end_ts(CLOSE, 45) == int(CLOSE.timestamp()) - 15 * 60
    assert checkpoint_end_ts(CLOSE, 40) == int(CLOSE.timestamp()) - 20 * 60
    assert checkpoint_end_ts(int(CLOSE.timestamp()), 59) == int(CLOSE.timestamp()) - 60
    with pytest.raises(ValueError, match="1..59"):
        checkpoint_end_ts(CLOSE, 0)
    with pytest.raises(ValueError, match="1..59"):
        checkpoint_end_ts(CLOSE, 60)


def test_evaluate_yes_win_and_no_complement(db) -> None:
    store_settled_events(db, [_settled(expiration="150", winner="mid")])
    end_ts = checkpoint_end_ts(CLOSE, 45)
    store_btc_candles(db, [_bar(end_ts, "150")])

    yes = evaluate(db, minutes=(45,), side="yes")
    no = evaluate(db, minutes=(45,), side="no")
    assert yes.minutes[0].eligible == 1
    assert yes.minutes[0].wins == 1
    assert yes.minutes[0].losses == 0
    assert no.minutes[0].wins == 0
    assert no.minutes[0].losses == 1


def test_evaluate_yes_loss_when_price_leaves_bracket(db) -> None:
    store_settled_events(db, [_settled(expiration="250", winner="high")])
    end_ts = checkpoint_end_ts(CLOSE, 50)
    store_btc_candles(db, [_bar(end_ts, "150")])

    report = evaluate(db, minutes=(50,), side="yes")
    assert report.minutes[0].wins == 0
    assert report.minutes[0].losses == 1


def test_evaluate_skips_missing_bar(db) -> None:
    store_settled_events(db, [_settled()])
    report = evaluate(db, minutes=(45,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].exclusions == {"missing_bar": 1}


def test_evaluate_date_bounds_half_open(db) -> None:
    early = CLOSE
    late = CLOSE + timedelta(hours=1)
    store_settled_events(
        db,
        [
            _settled(
                event_ticker="KXBTC-99JUN0108",
                close_ts=early,
                expiration="150",
            ),
            _settled(
                event_ticker="KXBTC-99JUN0109",
                close_ts=late,
                expiration="150",
            ),
        ],
    )
    for close_ts in (early, late):
        store_btc_candles(
            db,
            [_bar(checkpoint_end_ts(close_ts, 45), "150")],
        )

    report = evaluate(
        db,
        minutes=(45,),
        side="yes",
        start=early,
        end=late,
    )
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1


def test_evaluate_bad_winner_invariant(db) -> None:
    # Settlement 150 is in MID, but HIGH is marked won.
    store_settled_events(db, [_settled(expiration="150", winner="high")])
    store_btc_candles(db, [_bar(checkpoint_end_ts(CLOSE, 45), "150")])
    report = evaluate(db, minutes=(45,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].exclusions == {"bad_winner_invariant": 1}


def test_parse_checkpoint_minutes() -> None:
    assert parse_checkpoint_minutes("40,45,50") == (40, 45, 50)
    with pytest.raises(typer.BadParameter, match="1..59"):
        parse_checkpoint_minutes("0,45")
    with pytest.raises(typer.BadParameter, match="unique"):
        parse_checkpoint_minutes("45,45")


def test_evaluate_rejects_duplicate_minutes(db) -> None:
    with pytest.raises(ValueError, match="unique"):
        evaluate(db, minutes=(45, 45), side="yes")


def test_load_events_fractional_second_bounds(db) -> None:
    store_settled_events(db, [_settled(close_ts=CLOSE)])
    included = load_events(
        db,
        CLOSE - timedelta(seconds=0.5),
        CLOSE + timedelta(milliseconds=500),
    )
    assert [event.event_ticker for event in included] == ["KXBTC-99JUN0108"]
    excluded_by_start = load_events(
        db,
        CLOSE + timedelta(milliseconds=500),
        CLOSE + timedelta(hours=1),
    )
    assert excluded_by_start == ()
    excluded_by_end = load_events(
        db,
        CLOSE - timedelta(hours=1),
        CLOSE - timedelta(milliseconds=500),
    )
    assert excluded_by_end == ()


def test_evaluate_rejects_nonfinite_expiration(db) -> None:
    store_settled_events(db, [_settled(expiration="150", winner="mid")])
    db.execute(
        "UPDATE events SET expiration_value = ? WHERE event_ticker = ?",
        ("NaN", "KXBTC-99JUN0108"),
    )
    db.commit()
    store_btc_candles(db, [_bar(checkpoint_end_ts(CLOSE, 45), "150")])
    report = evaluate(db, minutes=(45,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].exclusions == {"bad_expiration": 1}


def test_evaluate_rejects_nonfinite_candle_close(db) -> None:
    store_settled_events(db, [_settled(expiration="150", winner="mid")])
    store_btc_candles(db, [_bar(checkpoint_end_ts(CLOSE, 45), "Infinity")])
    report = evaluate(db, minutes=(45,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].exclusions == {"bad_price": 1}


def test_price_in_bracket_bad_strikes() -> None:
    assert not price_in_bracket(Decimal("150"), "NaN", "200")
    assert not price_in_bracket(Decimal("150"), "100", "nope")
    assert not price_in_bracket(Decimal("NaN"), "100", "200")


def test_evaluate_no_bracket_match(db) -> None:
    event_ticker = "KXBTC-99JUN0108"
    store_settled_events(
        db,
        [
            SettledEvent(
                event=Event(
                    event_ticker=event_ticker,
                    close_ts=CLOSE,
                    status=STATUS_COMPLETE,
                    expiration_value=Decimal("150"),
                ),
                brackets=(
                    Bracket(
                        ticker=f"{event_ticker}-MID",
                        event_ticker=event_ticker,
                        floor_strike=Decimal("100"),
                        cap_strike=Decimal("199.99"),
                        won=True,
                    ),
                ),
            )
        ],
    )
    store_btc_candles(db, [_bar(checkpoint_end_ts(CLOSE, 45), "50")])
    report = evaluate(db, minutes=(45,), side="yes")
    assert report.minutes[0].exclusions == {"no_bracket_match": 1}


def test_evaluate_reports_strategy_name(db) -> None:
    store_settled_events(db, [_settled()])
    store_btc_candles(db, [_bar(checkpoint_end_ts(CLOSE, 45), "150")])
    report = evaluate(db, minutes=(45,), side="yes")
    assert report.strategy == STRATEGY == "s1"


def test_research_s1_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [_settled()])
    store_btc_candles(connection, [_bar(checkpoint_end_ts(CLOSE, 45), "150")])
    connection.close()

    result = runner.invoke(
        app,
        [
            "research",
            "s1",
            "--db",
            str(db_path),
            "--minutes",
            "45",
            "--side",
            "yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "s1" in result.output
    assert "win_rate" in result.output
    assert "100.0%" in result.output
