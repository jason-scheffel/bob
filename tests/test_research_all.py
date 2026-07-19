# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from bob.db import (
    MinuteBar,
    connect,
    initialize_schema,
    store_btc_candles,
    store_settled_events,
)
from bob.kalshi import STATUS_COMPLETE, Bracket, Event, SettledEvent
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


def test_strategy_names_cover_s1_to_s13() -> None:
    assert STRATEGY_NAMES == tuple(f"s{i}" for i in range(1, 14))


def test_run_all_rejects_memory_db() -> None:
    with pytest.raises(ValueError, match="on-disk"):
        run_all_strategies(":memory:", minutes=(50,), side="yes")
    with pytest.raises(ValueError, match="on-disk"):
        run_all_strategy_pnl(":memory:", minutes=(50,), side="yes")
