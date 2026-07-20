# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.cli import app
from bob.db import (
    MinuteBar,
    connect,
    initialize_schema,
    store_btc_candles,
    store_settled_events,
)
from bob.research.common import checkpoint_end_ts
from bob.research.s11 import STRATEGY, confirmation_minute, evaluate, replay_window
from helpers import RESEARCH_CLOSE, research_settled, seed_research_yes_quote

runner = CliRunner()


@pytest.fixture
def db():
    connection = connect(":memory:")
    initialize_schema(connection)
    yield connection
    connection.close()


def _seed_replay(
    connection,
    *,
    anchor_close: str = "150",
    window_high: str = "160",
    window_low: str = "140",
    checkpoint_close: str = "150",
) -> None:
    # Anchor at M55 is minute 50; replay window is 51..55.
    bars = [
        MinuteBar(
            end_ts=checkpoint_end_ts(RESEARCH_CLOSE, 50),
            open=anchor_close,
            high=anchor_close,
            low=anchor_close,
            close=anchor_close,
        )
    ]
    for minute in replay_window(55):
        close = checkpoint_close if minute == 55 else "150"
        bars.append(
            MinuteBar(
                end_ts=checkpoint_end_ts(RESEARCH_CLOSE, minute),
                open="150",
                high=window_high,
                low=window_low,
                close=close,
            )
        )
    store_btc_candles(connection, bars)


def test_confirmation_and_replay_window() -> None:
    assert confirmation_minute(45) == 30
    assert confirmation_minute(55) == 50
    assert replay_window(45) == range(31, 46)
    assert replay_window(55) == range(51, 56)
    with pytest.raises(ValueError, match="31..59"):
        confirmation_minute(30)


def test_s11_takes_trade_when_replay_fits(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    _seed_replay(db)

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.strategy == STRATEGY == "s11"
    assert report.minutes[0].eligible == 1
    assert report.minutes[0].wins == 1
    assert report.minutes[0].abstained == 0


def test_s11_abstains_on_up_breach(db) -> None:
    store_settled_events(db, [research_settled(expiration="150", winner="a")])
    # Anchor 150, high 210 → u=60; at P=150 bracket A cap~200 leaves only 50 room.
    _seed_replay(db, window_high="210")

    report = evaluate(db, minutes=(55,), side="yes")

    assert report.minutes[0].eligible == 0
    assert report.minutes[0].abstentions == {"replay_breach_up": 1}


def test_research_s11_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    _seed_replay(connection)
    seed_research_yes_quote(connection, ticker="KXBTC-99JUN0108-A", minute=55)
    connection.close()

    result = runner.invoke(
        app,
        ["research", "s11", "--db", str(db_path), "--minutes", "55"],
    )

    assert result.exit_code == 0, result.output
    assert "s11" in result.output
    assert "100.0%" in result.output
    assert "abstained" in result.output
