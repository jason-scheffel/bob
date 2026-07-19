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
