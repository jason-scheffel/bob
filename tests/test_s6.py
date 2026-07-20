# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from bob.db import (
    MinuteBar,
    connect,
    initialize_schema,
    store_btc_candles,
    store_settled_events,
)
from bob.research.common import checkpoint_end_ts
from bob.research.s6 import evaluate
from helpers import RESEARCH_CLOSE, research_flat_bars, research_settled


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def _rising_breakout() -> list[MinuteBar]:
    bars = research_flat_bars(range(1, 53), "150")
    for minute, price in ((53, "170"), (54, "180"), (55, "190")):
        text = str(price)
        bars.append(
            MinuteBar(
                end_ts=checkpoint_end_ts(RESEARCH_CLOSE, minute),
                open=text,
                high=text,
                low=text,
                close=text,
            )
        )
    return bars


def test_s6_breakout_to_neighbor(db) -> None:
    # In upper 20% of A with rising closes → buy B; B wins.
    store_settled_events(db, [research_settled(expiration="250", winner="b")])
    store_btc_candles(db, _rising_breakout())
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1
    assert len(report.trades) == 1
    assert report.trades[0].market_ticker.endswith("-B")
    assert report.trades[0].end_ts == checkpoint_end_ts(RESEARCH_CLOSE, 55)
    assert report.trades[0].won is True


def test_s6_abstains_no_direction_on_flat(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, research_flat_bars(range(1, 56), "150"))
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("no_direction", 0) == 1


def test_s6_abstains_not_near_edge(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 53), "150")
    for minute, price in ((53, "151"), (54, "152"), (55, "153")):
        text = str(price)
        bars.append(
            MinuteBar(
                end_ts=checkpoint_end_ts(RESEARCH_CLOSE, minute),
                open=text,
                high=text,
                low=text,
                close=text,
            )
        )
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("not_near_edge", 0) == 1


def test_s6_loss_when_neighbor_loses(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, _rising_breakout())
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].losses == 1
    assert report.trades[0].market_ticker.endswith("-B")


def test_s6_no_side_on_neighbor(db) -> None:
    store_settled_events(db, [research_settled(expiration="250", winner="b")])
    store_btc_candles(db, _rising_breakout())
    report = evaluate(db, minutes=(55,), side="no")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].losses == 1
