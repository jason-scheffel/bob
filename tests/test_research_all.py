# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

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
from bob.research.runner import (
    STRATEGY_NAMES,
    run_all_strategies,
    run_all_strategy_pnl,
    run_strategy_on_db,
    run_strategy_pnl_on_db,
)

CLOSE = datetime(2099, 6, 1, 12, 0, tzinfo=timezone.utc)


def _seed(db_path: Path) -> None:
    connection = connect(db_path)
    initialize_schema(connection)
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
    connection.close()


def test_research_all_matches_solo_and_order(tmp_path: Path) -> None:
    db_path = tmp_path / "bob.sqlite"
    _seed(db_path)
    solo = {
        name: run_strategy_on_db(
            db_path, name, minutes=(50,), side="yes", readonly=True
        )
        for name in STRATEGY_NAMES
    }
    combined = run_all_strategies(
        db_path,
        minutes=(50,),
        side="yes",
        workers=2,
    )
    assert tuple(item.strategy for item in combined) == STRATEGY_NAMES
    for item in combined:
        assert item.eligible == solo[item.strategy].eligible
        assert item.wins == solo[item.strategy].wins
        assert item.losses == solo[item.strategy].losses
        assert item.win_rate == solo[item.strategy].win_rate


def test_research_pnl_all_matches_solo(tmp_path: Path) -> None:
    db_path = tmp_path / "bob.sqlite"
    _seed(db_path)
    names = ("s1", "s2", "s10")
    solo = {
        name: run_strategy_pnl_on_db(
            db_path, name, minutes=(50,), side="yes", readonly=True
        )
        for name in names
    }
    combined = run_all_strategy_pnl(
        db_path,
        minutes=(50,),
        side="yes",
        workers=2,
        names=names,
    )
    assert tuple(item.strategy for item in combined) == names
    for item in combined:
        assert item.pnl.strategy_eligible == solo[item.strategy].pnl.strategy_eligible
        assert item.pnl.quote_eligible == solo[item.strategy].pnl.quote_eligible
        assert item.pnl.gross == solo[item.strategy].pnl.gross
        assert item.by_minute == solo[item.strategy].by_minute


def test_strategy_names_cover_s1_to_s23() -> None:
    assert STRATEGY_NAMES == tuple(f"s{i}" for i in range(1, 24))


def test_run_all_rejects_memory_db() -> None:
    with pytest.raises(ValueError, match="on-disk"):
        run_all_strategies(":memory:", minutes=(50,), side="yes")
    with pytest.raises(ValueError, match="on-disk"):
        run_all_strategy_pnl(":memory:", minutes=(50,), side="yes")


def _seed_s1_with_exit_quotes(db_path: Path) -> None:
    """Synthetic-looking path: flat mid band + entry/exit quotes for overlays."""
    connection = connect(db_path)
    initialize_schema(connection)
    event_ticker = "SYNTH-EVENT-ALL-001"
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
                        ticker=f"{event_ticker}-BAND-A",
                        event_ticker=event_ticker,
                        floor_strike=Decimal("100"),
                        cap_strike=Decimal("199.99"),
                        won=True,
                    ),
                    Bracket(
                        ticker=f"{event_ticker}-BAND-B",
                        event_ticker=event_ticker,
                        floor_strike=Decimal("200"),
                        cap_strike=Decimal("299.99"),
                        won=False,
                    ),
                    Bracket(
                        ticker=f"{event_ticker}-BAND-C",
                        event_ticker=event_ticker,
                        floor_strike=Decimal("300"),
                        cap_strike=Decimal("399.99"),
                        won=False,
                    ),
                ),
            )
        ],
    )
    store_btc_candles(
        connection,
        [
            MinuteBar(
                end_ts=checkpoint_end_ts(CLOSE, minute),
                open="150",
                high="150",
                low="150",
                close="150",
            )
            for minute in range(1, 56)
        ],
    )
    ticker = f"{event_ticker}-BAND-A"
    store_market_candles(
        connection,
        [
            MarketQuoteBar(
                ticker=ticker,
                end_ts=checkpoint_end_ts(CLOSE, 55),
                yes_bid_close="0.50",
                yes_ask_close="0.55",
            ),
            MarketQuoteBar(
                ticker=ticker,
                end_ts=checkpoint_end_ts(CLOSE, 56),
                yes_bid_close="0.20",
                yes_ask_close="0.22",
            ),
            MarketQuoteBar(
                ticker=ticker,
                end_ts=checkpoint_end_ts(CLOSE, 57),
                yes_bid_close="0.70",
                yes_ask_close="0.75",
            ),
        ],
    )
    connection.close()


@pytest.mark.parametrize(
    ("stop_bid", "take_pct", "stop_from", "stopped", "taken"),
    [
        (None, None, 55, 0, 0),
        (Decimal("0.30"), None, 55, 1, 0),
        (None, Decimal("0.20"), 55, 0, 1),
        (Decimal("0.30"), Decimal("0.20"), 55, 1, 0),  # stop before take
    ],
)
def test_run_strategy_pnl_overlays_on_synth(
    tmp_path: Path,
    stop_bid: Decimal | None,
    take_pct: Decimal | None,
    stop_from: int,
    stopped: int,
    taken: int,
) -> None:
    db_path = tmp_path / "bob.sqlite"
    _seed_s1_with_exit_quotes(db_path)
    solo = run_strategy_pnl_on_db(
        db_path,
        "s1",
        minutes=(55,),
        side="yes",
        stop_bid=stop_bid,
        take_pct=take_pct,
        stop_from=stop_from,
        readonly=True,
    )
    assert solo.pnl.quote_eligible == 1
    assert solo.pnl.stopped == stopped
    assert solo.pnl.taken == taken
    if stopped:
        trade = solo.pnl.trades[0]
        assert trade.exit_minute == 56
        assert trade.settlement == Decimal("0.20")
    if taken and not stopped:
        trade = solo.pnl.trades[0]
        assert trade.exit_minute == 57
        assert trade.settlement == Decimal("0.70")

    combined = run_all_strategy_pnl(
        db_path,
        minutes=(55,),
        side="yes",
        workers=1,
        names=("s1",),
        stop_bid=stop_bid,
        take_pct=take_pct,
        stop_from=stop_from,
    )
    assert len(combined) == 1
    assert combined[0].pnl.stopped == solo.pnl.stopped
    assert combined[0].pnl.taken == solo.pnl.taken
    assert combined[0].pnl.gross == solo.pnl.gross
