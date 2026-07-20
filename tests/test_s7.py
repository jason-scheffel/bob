# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from bob.db import connect, initialize_schema, store_btc_candles, store_settled_events
from bob.research.s7 import evaluate
from helpers import research_flat_bars, research_settled


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_s7_modal_bracket(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 50), "150")
    bars.extend(research_flat_bars(range(50, 56), "250"))
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1
    assert report.trades[0].market_ticker.endswith("-A")


def test_s7_abstains_no_unique_mode(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 21), "150")
    bars.extend(research_flat_bars(range(21, 41), "250"))
    bars.extend(research_flat_bars(range(41, 56), "350"))
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"no_unique_mode": 1}


def test_s7_abstains_low_occupancy(db) -> None:
    # Unique mode B with only 20/55 < ceil(55/2)=28
    store_settled_events(db, [research_settled(expiration="250", winner="b")])
    bars = research_flat_bars(range(1, 18), "150")
    bars.extend(research_flat_bars(range(18, 38), "250"))
    bars.extend(research_flat_bars(range(38, 56), "350"))
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions.get("low_occupancy", 0) == 1


def test_s7_loss_when_modal_bracket_loses(db) -> None:
    store_settled_events(db, [research_settled(expiration="250", winner="b")])
    bars = research_flat_bars(range(1, 50), "150")
    bars.extend(research_flat_bars(range(50, 56), "250"))
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].losses == 1


def test_s7_no_side_complements(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(1, 50), "150")
    bars.extend(research_flat_bars(range(50, 56), "250"))
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(55,), side="no")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].losses == 1
