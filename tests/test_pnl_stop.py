# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.cli import app
from bob.db import (
    MarketQuoteBar,
    connect,
    initialize_schema,
    store_btc_candles,
    store_market_candles,
    store_settled_events,
)
from bob.research.common import checkpoint_end_ts
from bob.research.pnl import score_trades
from bob.research.trades import TradeObservation
from helpers import RESEARCH_CLOSE, research_flat_bars, research_settled

CLOSE = datetime(2099, 6, 1, 12, 0, tzinfo=timezone.utc)
EVENT = "KXBTC-99JUN0108"
TICKER = f"{EVENT}-A"
runner = CliRunner()


def _obs(*, minute: int = 45, won: bool = True, side: str = "yes") -> TradeObservation:
    return TradeObservation(
        event_ticker=EVENT,
        market_ticker=TICKER,
        minute=minute,
        end_ts=checkpoint_end_ts(CLOSE, minute),
        side=side,  # type: ignore[arg-type]
        won=won,
    )


def _quote(minute: int, *, bid: str, ask: str) -> MarketQuoteBar:
    return MarketQuoteBar(
        ticker=TICKER,
        end_ts=checkpoint_end_ts(CLOSE, minute),
        yes_bid_close=bid,
        yes_ask_close=ask,
    )


def test_no_stop_bid_matches_hold_to_settlement() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.50", ask="0.55"),
            _quote(55, bid="0.20", ask="0.25"),
        ],
    )
    obs = _obs(minute=45, won=True)
    plain = score_trades(connection, [obs])
    with_from = score_trades(
        connection, [obs], stop_bid=None, stop_from=55
    )
    assert plain.gross == with_from.gross == Decimal("0.45")
    assert plain.stopped == with_from.stopped == 0
    assert plain.trades[0].stopped is False
    connection.close()


def test_stop_exits_at_first_breach_from_stop_from() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.50", ask="0.55"),
            _quote(50, bid="0.20", ask="0.25"),
            _quote(55, bid="0.28", ask="0.32"),
            _quote(56, bid="0.25", ask="0.30"),
        ],
    )
    obs = _obs(minute=45, won=True)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        stop_from=55,
    )
    trade = pnl.trades[0]
    assert trade.stopped is True
    assert trade.exit_minute == 55
    assert trade.settlement == Decimal("0.28")
    assert trade.gross == Decimal("0.28") - Decimal("0.55")
    assert pnl.stopped == 1
    assert pnl.wins == 0
    assert pnl.losses == 1
    connection.close()


def test_breach_before_stop_from_is_ignored() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.50", ask="0.55"),
            _quote(50, bid="0.10", ask="0.15"),
            _quote(56, bid="0.60", ask="0.65"),
        ],
    )
    obs = _obs(minute=45, won=True)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        stop_from=55,
    )
    assert pnl.trades[0].stopped is False
    assert pnl.trades[0].settlement == Decimal("1")
    assert pnl.gross == Decimal("0.45")
    connection.close()


def test_sparse_bars_skipped_until_breach() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.50", ask="0.55"),
            MarketQuoteBar(
                ticker=TICKER,
                end_ts=checkpoint_end_ts(CLOSE, 55),
                yes_bid_close=None,
                yes_ask_close=None,
            ),
            _quote(57, bid="0.22", ask="0.24"),
        ],
    )
    obs = _obs(minute=45, won=False)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        stop_from=55,
    )
    trade = pnl.trades[0]
    assert trade.stopped is True
    assert trade.exit_minute == 57
    assert trade.settlement == Decimal("0.22")
    connection.close()


def test_no_breach_holds_to_settlement() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.50", ask="0.55"),
            _quote(55, bid="0.40", ask="0.45"),
            _quote(59, bid="0.35", ask="0.40"),
        ],
    )
    obs = _obs(minute=45, won=False)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        stop_from=55,
    )
    assert pnl.trades[0].stopped is False
    assert pnl.trades[0].settlement == Decimal("0")
    assert pnl.gross == Decimal("-0.55")
    connection.close()


def test_no_side_stops_on_one_minus_ask() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.40", ask="0.45"),
            # NO mark = 1 - ask; ask 0.75 → mark 0.25 ≤ 0.30
            _quote(55, bid="0.70", ask="0.75"),
        ],
    )
    obs = _obs(minute=45, won=True, side="no")
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        stop_from=55,
    )
    trade = pnl.trades[0]
    assert trade.premium == Decimal("0.60")
    assert trade.stopped is True
    assert trade.settlement == Decimal("0.25")
    connection.close()


def test_invalid_stop_params_raise() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    with pytest.raises(ValueError, match="stop_bid"):
        score_trades(connection, [], stop_bid=Decimal("1.5"))
    with pytest.raises(ValueError, match="stop_from"):
        score_trades(
            connection, [], stop_bid=Decimal("0.3"), stop_from=0
        )
    connection.close()


def test_research_s21_cli_accepts_stop_bid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    store_btc_candles(connection, research_flat_bars(range(1, 56), "150"))
    store_market_candles(
        connection,
        [
            MarketQuoteBar(
                ticker="KXBTC-99JUN0108-A",
                end_ts=checkpoint_end_ts(RESEARCH_CLOSE, 55),
                yes_bid_close="0.50",
                yes_ask_close="0.55",
            )
        ],
    )
    connection.close()

    result = runner.invoke(
        app,
        [
            "research",
            "s21",
            "--db",
            str(db_path),
            "--minutes",
            "55",
            "--stop-bid",
            "0.30",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "stop-bid≤0.30" in result.output
    assert "stopped" in result.output
