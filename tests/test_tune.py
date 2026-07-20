# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from decimal import Decimal
from pathlib import Path

import optuna
import pytest
from rich.console import Console
from typer.testing import CliRunner

from bob.cli import app
from bob.db import (
    MarketQuoteBar,
    MinuteBar,
    connect,
    initialize_schema,
    store_btc_candles,
    store_market_candles,
    store_settled_events,
)
from bob.kalshi import STATUS_COMPLETE, Bracket, Event, SettledEvent
from bob.research.common import checkpoint_end_ts
from bob.research.tune import (
    TuneConfig,
    count_complete_events,
    evaluate_trial,
    min_eligible_trades,
    parse_strategy_names,
    run_tune,
)
from helpers import RESEARCH_CLOSE

runner = CliRunner()


def test_parse_strategy_names() -> None:
    assert parse_strategy_names("s20,s21") == ("s20", "s21")
    with pytest.raises(ValueError, match="unknown"):
        parse_strategy_names("s99")
    with pytest.raises(ValueError, match="unique"):
        parse_strategy_names("s20,s20")


def test_min_eligible_trades_ceil() -> None:
    assert min_eligible_trades(n_events=720, min_frac=Decimal("0.25")) == 180
    assert min_eligible_trades(n_events=10, min_frac=Decimal("0.25")) == 3


def _seed_s1_db(db_path: Path, *, n_events: int = 4) -> None:
    connection = connect(db_path)
    initialize_schema(connection)
    for offset in range(n_events):
        close = RESEARCH_CLOSE.replace(hour=12 + offset)
        ticker = f"KXBTC-99JUN{offset:02d}08"
        settled = SettledEvent(
            event=Event(
                event_ticker=ticker,
                close_ts=close,
                status=STATUS_COMPLETE,
                expiration_value=Decimal("150"),
            ),
            brackets=(
                Bracket(
                    ticker=f"{ticker}-A",
                    event_ticker=ticker,
                    floor_strike=Decimal("100"),
                    cap_strike=Decimal("199.99"),
                    won=True,
                ),
                Bracket(
                    ticker=f"{ticker}-B",
                    event_ticker=ticker,
                    floor_strike=Decimal("200"),
                    cap_strike=Decimal("299.99"),
                    won=False,
                ),
                Bracket(
                    ticker=f"{ticker}-C",
                    event_ticker=ticker,
                    floor_strike=Decimal("300"),
                    cap_strike=Decimal("399.99"),
                    won=False,
                ),
            ),
        )
        store_settled_events(connection, [settled])
        bars = [
            MinuteBar(
                end_ts=checkpoint_end_ts(close, minute),
                open="150",
                high="150",
                low="150",
                close="150",
            )
            for minute in range(1, 56)
        ]
        store_btc_candles(connection, bars)
        store_market_candles(
            connection,
            [
                MarketQuoteBar(
                    ticker=f"{ticker}-A",
                    end_ts=checkpoint_end_ts(close, 55),
                    yes_bid_close="0.40",
                    yes_ask_close="0.45",
                )
            ],
        )
    connection.close()


def test_count_complete_events_and_n_floor_reject(tmp_path: Path) -> None:
    db_path = tmp_path / "bob.sqlite"
    _seed_s1_db(db_path, n_events=4)
    assert count_complete_events(db_path, start=None, end=None) == 4

    config = TuneConfig(
        db=db_path,
        strategies=("s1",),
        minutes=(55,),
        side="yes",
        start=None,
        end=None,
        min_frac=Decimal("0.90"),  # need ceil(0.9*4)=4; s1 may trade all 4
        n_trials=1,
        n_jobs=1,
    )
    # Force reject with min_n above possible
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    # Monkeypatch by evaluating with absurd min_n via direct call
    value = evaluate_trial(trial, config=config, min_n=10_000)
    assert value == float("-inf")
    assert trial.user_attrs.get("reject") == "n_floor"


def test_evaluate_trial_accepts_and_records_overlays(tmp_path: Path) -> None:
    db_path = tmp_path / "bob.sqlite"
    _seed_s1_db(db_path, n_events=4)
    config = TuneConfig(
        db=db_path,
        strategies=("s1",),
        minutes=(55,),
        side="yes",
        start=None,
        end=None,
        min_frac=Decimal("0.25"),
        n_trials=1,
        n_jobs=1,
    )
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        # Pin overlays for determinism via enqueue — use suggest path normally
        return evaluate_trial(trial, config=config, min_n=1)

    study.optimize(objective, n_trials=3, show_progress_bar=False)
    complete = [t for t in study.trials if t.value is not None and t.value > float("-inf")]
    assert complete
    assert complete[0].user_attrs["strategy"] == "s1"
    assert complete[0].user_attrs["minute"] == 55


def test_run_tune_smoke(tmp_path: Path) -> None:
    db_path = tmp_path / "bob.sqlite"
    _seed_s1_db(db_path, n_events=4)
    result = run_tune(
        TuneConfig(
            db=db_path,
            strategies=("s1",),
            minutes=(55,),
            side="yes",
            start=None,
            end=None,
            min_frac=Decimal("0.25"),
            n_trials=3,
            n_jobs=1,
            top_k=3,
        ),
        console=Console(stderr=True, force_terminal=False),
    )
    assert result.n_events == 4
    assert result.min_n == 1
    assert len(result.study.trials) == 3


def test_research_tune_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    _seed_s1_db(db_path, n_events=4)

    result = runner.invoke(
        app,
        [
            "research",
            "tune",
            "--db",
            str(db_path),
            "--strategies",
            "s1",
            "--minutes",
            "55",
            "--trials",
            "2",
            "--min-frac",
            "0.25",
        ],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert "events=4" in result.output
    assert "tune" in result.output.lower() or "top trials" in result.output
