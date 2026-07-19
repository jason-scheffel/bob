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
from bob.research.s5 import evaluate
from helpers import RESEARCH_CLOSE, research_settled


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_s5_thin_buffer_abstains(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    bars = []
    prices = [150, 190, 120, 190, 120, 190, 120, 190, 120, 190, 195]
    for price, minute in zip(prices, range(45, 56), strict=True):
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
    assert report.minutes[0].abstentions.get("thin_buffer", 0) == 1
