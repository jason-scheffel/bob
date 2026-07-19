# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone
from decimal import Decimal

from bob.db import (
    MarketQuoteBar,
    MinuteBar,
    connect,
    initialize_schema,
    store_btc_candles,
    store_market_candles,
    store_settled_events,
)
from bob.kalshi import STATUS_COMPLETE, Bracket, Event, SettledEvent
from bob.research.common import checkpoint_end_ts
from bob.research.pnl import score_trades, score_trades_by_minute, validate_quote
from bob.research.s1 import evaluate
from bob.research.trades import TradeObservation

CLOSE = datetime(2099, 6, 1, 12, 0, tzinfo=timezone.utc)
EVENT = "KXBTC-99JUN0108"
TICKER = f"{EVENT}-MID"


def test_validate_quote_rejects_crossed_and_out_of_range() -> None:
    assert validate_quote("0.40", "0.45") == (Decimal("0.40"), Decimal("0.45"))
    assert validate_quote(None, "0.45") is None
    assert validate_quote("0.50", "0.40") is None
    assert validate_quote("-0.01", "0.40") is None
    assert validate_quote("0.40", "1.01") is None


def _seed_outcome_and_quotes(connection, *, bid: str, ask: str) -> None:
    store_settled_events(
        connection,
        [
            SettledEvent(
                event=Event(
                    event_ticker=EVENT,
                    close_ts=CLOSE,
                    status=STATUS_COMPLETE,
                    expiration_value=Decimal("420"),
                ),
                brackets=(
                    Bracket(
                        ticker=TICKER,
                        event_ticker=EVENT,
                        floor_strike=Decimal("400"),
                        cap_strike=Decimal("499.99"),
                        won=True,
                    ),
                    Bracket(
                        ticker=f"{EVENT}-LOW",
                        event_ticker=EVENT,
                        floor_strike=None,
                        cap_strike=Decimal("399.99"),
                        won=False,
                    ),
                ),
            )
        ],
    )
    close_ts = int(CLOSE.timestamp())
    store_btc_candles(
        connection,
        [
            MinuteBar(
                end_ts=end_ts,
                open="420",
                high="421",
                low="419",
                close="420",
            )
            for end_ts in range(close_ts - 3540, close_ts + 1, 60)
        ],
    )
    end_ts = checkpoint_end_ts(CLOSE, 50)
    store_market_candles(
        connection,
        [
            MarketQuoteBar(
                ticker=TICKER,
                end_ts=end_ts,
                yes_bid_close=bid,
                yes_ask_close=ask,
            )
        ],
    )


def test_score_trades_yes_gross_and_return() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    _seed_outcome_and_quotes(connection, bid="0.40", ask="0.55")
    report = evaluate(connection, minutes=(50,), side="yes")
    pnl = score_trades(connection, report.trades)
    assert pnl.strategy_eligible == 1
    assert pnl.quote_eligible == 1
    assert pnl.premium == Decimal("0.55")
    assert pnl.payout == Decimal("1")
    assert pnl.gross == Decimal("0.45")
    assert pnl.return_on_premium == Decimal("0.45") / Decimal("0.55")
    assert pnl.win_rate == 1.0
    connection.close()


def test_score_trades_no_premium_uses_one_minus_bid() -> None:
    obs = TradeObservation(
        event_ticker=EVENT,
        market_ticker=TICKER,
        minute=50,
        end_ts=checkpoint_end_ts(CLOSE, 50),
        side="no",
        won=True,
    )
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            MarketQuoteBar(
                ticker=TICKER,
                end_ts=obs.end_ts,
                yes_bid_close="0.40",
                yes_ask_close="0.55",
            )
        ],
    )
    pnl = score_trades(connection, [obs])
    assert pnl.premium == Decimal("0.60")
    assert pnl.gross == Decimal("0.40")
    connection.close()


def test_score_trades_by_minute_splits() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    _seed_outcome_and_quotes(connection, bid="0.40", ask="0.55")
    end_50 = checkpoint_end_ts(CLOSE, 50)
    end_55 = checkpoint_end_ts(CLOSE, 55)
    store_market_candles(
        connection,
        [
            MarketQuoteBar(
                ticker=TICKER,
                end_ts=end_55,
                yes_bid_close="0.30",
                yes_ask_close="0.35",
            )
        ],
    )
    trades = (
        TradeObservation(
            event_ticker=EVENT,
            market_ticker=TICKER,
            minute=50,
            end_ts=end_50,
            side="yes",
            won=True,
        ),
        TradeObservation(
            event_ticker=EVENT,
            market_ticker=TICKER,
            minute=55,
            end_ts=end_55,
            side="yes",
            won=False,
        ),
    )
    by_minute = score_trades_by_minute(connection, trades, (50, 55))
    assert [minute for minute, _ in by_minute] == [50, 55]
    assert by_minute[0][1].premium == Decimal("0.55")
    assert by_minute[0][1].gross == Decimal("0.45")
    assert by_minute[1][1].premium == Decimal("0.35")
    assert by_minute[1][1].gross == Decimal("-0.35")
    connection.close()


def test_score_trades_excludes_bad_quote() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    _seed_outcome_and_quotes(connection, bid="0.70", ask="0.60")
    report = evaluate(connection, minutes=(50,), side="yes")
    pnl = score_trades(connection, report.trades)
    assert pnl.strategy_eligible == 1
    assert pnl.quote_excluded == 1
    assert pnl.quote_eligible == 0
    assert pnl.premium == 0
    connection.close()
