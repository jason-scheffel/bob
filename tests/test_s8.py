# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from bob.db import (
    connect,
    initialize_schema,
    store_btc_candles,
    store_settled_events,
)
from bob.research.s8 import confirmation_minute, evaluate
from helpers import research_flat_bars, research_settled


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_s8_horizon_confirmed_bracket_wins(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, research_flat_bars(range(50, 56), "150"))

    report = evaluate(db, minutes=(55,), side="yes")

    assert confirmation_minute(55) == 50
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1


def test_s8_unconfirmed_bracket_abstains(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = research_flat_bars(range(50, 51), "250")
    bars.extend(research_flat_bars(range(55, 56), "150"))
    store_btc_candles(db, bars)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"unconfirmed_bracket": 1}
