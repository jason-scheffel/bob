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


def _reversion_bars(*, open_px: str, checkpoint: str) -> list[MinuteBar]:
    bars = [
        MinuteBar(
            end_ts=checkpoint_end_ts(RESEARCH_CLOSE, 1),
            open=open_px,
            high=open_px,
            low=open_px,
            close=open_px,
        )
    ]
    for minute in range(2, 56):
        bars.append(
            MinuteBar(
                end_ts=checkpoint_end_ts(RESEARCH_CLOSE, minute),
                open=checkpoint,
                high=checkpoint,
                low=checkpoint,
                close=checkpoint,
            )
        )
    return bars


def test_s4_small_move_abstains(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, research_flat_bars(range(1, 56), "150"))
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].abstentions == {"small_move": 1}


def test_s4_boundary_move_249_abstains(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, _reversion_bars(open_px="100", checkpoint="349"))
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"small_move": 1}


def test_s4_boundary_move_250_trades(db) -> None:
    store_settled_events(db, [research_settled(expiration="250", winner="b")])
    store_btc_candles(db, _reversion_bars(open_px="100", checkpoint="350"))
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1


def test_s4_reverts_toward_open(db) -> None:
    # Open 100, checkpoint 400 → target 250 → bracket B; settle B.
    store_settled_events(db, [research_settled(expiration="250", winner="b")])
    store_btc_candles(db, _reversion_bars(open_px="100", checkpoint="400"))
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1
    assert report.trades[0].market_ticker.endswith("-B")
    assert report.trades[0].won is True


def test_s4_loss_when_target_bracket_loses(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, _reversion_bars(open_px="100", checkpoint="400"))
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].losses == 1


def test_s4_no_side_complements_win(db) -> None:
    store_settled_events(db, [research_settled(expiration="250", winner="b")])
    store_btc_candles(db, _reversion_bars(open_px="100", checkpoint="400"))
    report = evaluate(db, minutes=(55,), side="no")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].losses == 1
