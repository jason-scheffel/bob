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
from bob.research.s3 import evaluate
from helpers import RESEARCH_CLOSE, research_settled


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_s3_projects_up_into_higher_bracket(db) -> None:
    # Rising ~10/min over last 10 bars → projects well above 300.
    store_settled_events(db, [research_settled(expiration="350", winner="c")])
    bars = []
    for offset, minute in enumerate(range(46, 56)):
        price = str(200 + offset * 10)
        bars.append(
            MinuteBar(
                end_ts=checkpoint_end_ts(RESEARCH_CLOSE, minute),
                open=price,
                high=price,
                low=price,
                close=price,
            )
        )
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1
