# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Trade observation wiring must not change strategy win rates."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from bob.db import MinuteBar, connect, initialize_schema, store_btc_candles, store_settled_events
from bob.kalshi import STATUS_COMPLETE, Bracket, Event, SettledEvent
from bob.research import s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13
from bob.research.common import checkpoint_end_ts
from bob.research.runner import STRATEGY_MODULES, evaluate_strategy

CLOSE = datetime(2099, 6, 1, 12, 0, tzinfo=timezone.utc)


def _seed_flat_mid_winner(connection) -> None:
    event_ticker = "KXBTC-99JUN0108"
    store_settled_events(
        connection,
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
                        ticker=f"{event_ticker}-A",
                        event_ticker=event_ticker,
                        floor_strike=Decimal("100"),
                        cap_strike=Decimal("199.99"),
                        won=True,
                    ),
                    Bracket(
                        ticker=f"{event_ticker}-B",
                        event_ticker=event_ticker,
                        floor_strike=Decimal("200"),
                        cap_strike=Decimal("299.99"),
                        won=False,
                    ),
                    Bracket(
                        ticker=f"{event_ticker}-C",
                        event_ticker=event_ticker,
                        floor_strike=Decimal("300"),
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
                open="150",
                high="150",
                low="150",
                close="150",
            )
            for end_ts in range(close_ts - 3540, close_ts + 1, 60)
        ],
    )


@pytest.mark.parametrize(
    ("name", "minute"),
    [
        ("s1", 50),
        ("s2", 55),
        ("s3", 50),
        ("s4", 50),
        ("s5", 50),
        ("s7", 50),
        ("s8", 50),
        ("s9", 50),
        ("s10", 50),
        ("s11", 50),
        ("s12", 50),
        ("s13", 50),
    ],
)
def test_trade_count_matches_eligible(name: str, minute: int) -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    _seed_flat_mid_winner(connection)
    report = evaluate_strategy(connection, name, minutes=(minute,), side="yes")
    eligible = sum(stats.eligible for stats in report.minutes)
    wins = sum(stats.wins for stats in report.minutes)
    losses = sum(stats.losses for stats in report.minutes)
    assert len(report.trades) == eligible
    assert sum(1 for trade in report.trades if trade.won) == wins
    assert sum(1 for trade in report.trades if not trade.won) == losses
    for trade in report.trades:
        assert trade.end_ts == checkpoint_end_ts(CLOSE, trade.minute)
        assert trade.side == "yes"
    connection.close()


def test_s1_trades_match_eligible_and_preserve_win_rate() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    _seed_flat_mid_winner(connection)
    report = s1.evaluate(connection, minutes=(50,), side="yes")
    stats = report.minutes[0]
    assert stats.eligible == 1
    assert stats.wins == 1
    assert stats.win_rate == 1.0
    assert len(report.trades) == stats.eligible
    trade = report.trades[0]
    assert trade.minute == 50
    assert trade.end_ts == checkpoint_end_ts(CLOSE, 50)
    assert trade.won is True
    connection.close()


def test_exclusion_produces_no_trades() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    _seed_flat_mid_winner(connection)
    # Wipe candles → missing_bar exclusions, no trades.
    connection.execute("DELETE FROM btc_candles")
    connection.commit()
    report = s1.evaluate(connection, minutes=(50,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.trades == ()
    connection.close()


def test_strategy_modules_expose_trades_field() -> None:
    for name, module in STRATEGY_MODULES.items():
        assert "trades" in module.Report.__annotations__, name
