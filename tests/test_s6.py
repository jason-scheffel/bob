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


def test_s6_breakout_to_neighbor(db) -> None:
    # In upper 20% of A with rising closes → buy B; B wins.
    store_settled_events(db, [research_settled(expiration="250", winner="b")])
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
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1
