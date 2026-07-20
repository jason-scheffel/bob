# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from decimal import Decimal

import pytest

from bob.db import (
    MinuteBar,
    connect,
    initialize_schema,
    store_btc_candles,
    store_settled_events,
)
from bob.kalshi import STATUS_COMPLETE, Bracket, Event, SettledEvent
from bob.research.common import checkpoint_end_ts
from bob.research.s5 import evaluate
from helpers import RESEARCH_CLOSE, research_settled


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def _point_path(prices: list[int], start_minute: int = 45) -> list[MinuteBar]:
    return [
        MinuteBar(
            end_ts=checkpoint_end_ts(RESEARCH_CLOSE, minute),
            open=str(price),
            high=str(price),
            low=str(price),
            close=str(price),
        )
        for price, minute in zip(
            prices, range(start_minute, start_minute + len(prices)), strict=True
        )
    ]


def test_s5_trades_when_buffer_is_thick(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    prices = [150, 151, 150, 151, 150, 151, 150, 151, 150, 151, 150]
    store_btc_candles(db, _point_path(prices))
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1
    assert report.trades[0].market_ticker.endswith("-A")


def test_s5_thin_buffer_abstains(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    prices = [150, 190, 120, 190, 120, 190, 120, 190, 120, 190, 195]
    store_btc_candles(db, _point_path(prices))
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("thin_buffer", 0) == 1


def test_s5_open_ended_bracket_abstains(db) -> None:
    event_ticker = "KXBTC-99JUN0108"
    store_settled_events(
        db,
        [
            SettledEvent(
                event=Event(
                    event_ticker=event_ticker,
                    close_ts=RESEARCH_CLOSE,
                    status=STATUS_COMPLETE,
                    expiration_value=Decimal("250"),
                ),
                brackets=(
                    Bracket(
                        ticker=f"{event_ticker}-A",
                        event_ticker=event_ticker,
                        floor_strike=Decimal("100"),
                        cap_strike=Decimal("199.99"),
                        won=False,
                    ),
                    Bracket(
                        ticker=f"{event_ticker}-HIGH",
                        event_ticker=event_ticker,
                        floor_strike=Decimal("200"),
                        cap_strike=None,
                        won=True,
                    ),
                ),
            )
        ],
    )
    prices = [250, 251, 250, 251, 250, 251, 250, 251, 250, 251, 250]
    store_btc_candles(db, _point_path(prices))
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("open_ended_bracket", 0) == 1


def test_s5_loss_when_current_bracket_loses(db) -> None:
    store_settled_events(db, [research_settled(expiration="250", winner="b")])
    prices = [150, 151, 150, 151, 150, 151, 150, 151, 150, 151, 150]
    store_btc_candles(db, _point_path(prices))
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].losses == 1


def test_s5_no_side_complements(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    prices = [150, 151, 150, 151, 150, 151, 150, 151, 150, 151, 150]
    store_btc_candles(db, _point_path(prices))
    report = evaluate(db, minutes=(55,), side="no")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].losses == 1
