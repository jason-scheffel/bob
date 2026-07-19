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
from bob.research.s4 import evaluate
from helpers import RESEARCH_CLOSE, research_flat_bars, research_settled


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_s4_small_move_abstains(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, research_flat_bars(range(1, 56), "150"))
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].abstentions == {"small_move": 1}


def test_s4_reverts_toward_open(db) -> None:
    # Open 100, checkpoint 400 → target 250 → bracket B; settle B.
    store_settled_events(db, [research_settled(expiration="250", winner="b")])
    bars = [
        MinuteBar(
            end_ts=checkpoint_end_ts(RESEARCH_CLOSE, 1),
            open="100",
            high="100",
            low="100",
            close="100",
        )
    ]
    for minute in range(2, 56):
        bars.append(
            MinuteBar(
                end_ts=checkpoint_end_ts(RESEARCH_CLOSE, minute),
                open="400",
                high="400",
                low="400",
                close="400",
            )
        )
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1
