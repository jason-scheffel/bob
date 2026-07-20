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
from bob.research.common import checkpoint_end_ts, ols_slope
from bob.research.s3 import TREND_WINDOW, evaluate
from helpers import RESEARCH_CLOSE, research_settled


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def _rising_bars() -> list[MinuteBar]:
    return [
        MinuteBar(
            end_ts=checkpoint_end_ts(RESEARCH_CLOSE, minute),
            open=str(200 + offset * 10),
            high=str(200 + offset * 10),
            low=str(200 + offset * 10),
            close=str(200 + offset * 10),
        )
        for offset, minute in enumerate(range(46, 56))
    ]


def test_ols_slope_positive_on_rising_series() -> None:
    closes = [float(200 + i * 10) for i in range(TREND_WINDOW)]
    slope = ols_slope(closes)
    assert slope is not None
    assert slope == pytest.approx(10.0)


def test_ols_slope_zero_on_flat() -> None:
    assert ols_slope([150.0] * TREND_WINDOW) == pytest.approx(0.0)


def test_s3_projects_up_into_higher_bracket(db) -> None:
    # Rising ~10/min over last 10 bars → projects well above 300.
    store_settled_events(db, [research_settled(expiration="350", winner="c")])
    store_btc_candles(db, _rising_bars())
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1
    assert report.trades[0].market_ticker.endswith("-C")


def test_s3_no_side_loses_when_projected_bracket_wins(db) -> None:
    store_settled_events(db, [research_settled(expiration="350", winner="c")])
    store_btc_candles(db, _rising_bars())
    report = evaluate(db, minutes=(55,), side="no")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].losses == 1
    assert report.trades[0].won is False


def test_s3_loss_when_projected_bracket_loses(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    store_btc_candles(db, _rising_bars())
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].losses == 1


def test_s3_missing_bars_excluded(db) -> None:
    store_settled_events(db, [research_settled(expiration="350", winner="c")])
    # only one bar — not enough for trend window
    store_btc_candles(
        db,
        [
            MinuteBar(
                end_ts=checkpoint_end_ts(RESEARCH_CLOSE, 55),
                open="290",
                high="290",
                low="290",
                close="290",
            )
        ],
    )
    report = evaluate(db, minutes=(55,), side="yes")
    assert report.minutes[0].eligible == 0
    assert report.minutes[0].exclusions.get("missing_bars", 0) == 1


@pytest.mark.parametrize("minute", [50, 55])
def test_s3_accepts_default_checkpoint_minutes(db, minute: int) -> None:
    # Rising 10/min into ~290; remaining minutes project into band C.
    store_settled_events(db, [research_settled(expiration="350", winner="c")])
    bars = [
        MinuteBar(
            end_ts=checkpoint_end_ts(RESEARCH_CLOSE, m),
            open=str(200 + offset * 10),
            high=str(200 + offset * 10),
            low=str(200 + offset * 10),
            close=str(200 + offset * 10),
        )
        for offset, m in enumerate(range(minute - 9, minute + 1))
    ]
    store_btc_candles(db, bars)
    report = evaluate(db, minutes=(minute,), side="yes")
    assert report.minutes[0].eligible == 1
