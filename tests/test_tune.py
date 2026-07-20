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
from bob.research.pnl import QuoteSimTrade
from bob.research.trades import TradeObservation
from bob.research.tune import (
    TuneConfig,
    count_complete_events,
    evaluate_trial,
    min_eligible_trades,
    parse_strategy_names,
    run_tune,
    simulate_bankroll,
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


def _trade(
    *,
    end_ts: int,
    premium: str,
    settlement: str,
    event: str = "E",
) -> QuoteSimTrade:
    return QuoteSimTrade(
        observation=TradeObservation(
            event_ticker=event,
            market_ticker=f"{event}-A",
            minute=55,
            end_ts=end_ts,
            side="yes",
            won=settlement == "1",
        ),
        premium=Decimal(premium),
        settlement=Decimal(settlement),
        gross=Decimal(settlement) - Decimal(premium),
    )


def test_simulate_bankroll_ruin_on_loss_streak() -> None:
    # start 1.00; three full losses at 0.40 → cannot fund / dies
    trades = (
        _trade(end_ts=100, premium="0.40", settlement="0", event="A"),
        _trade(end_ts=200, premium="0.40", settlement="0", event="B"),
        _trade(end_ts=300, premium="0.40", settlement="0", event="C"),
    )
    survived, cash = simulate_bankroll(trades, start=Decimal("1.00"))
    assert survived is False
    assert cash < Decimal("0.40")


def test_simulate_bankroll_survives_and_tracks_cash() -> None:
    trades = (
        _trade(end_ts=100, premium="0.40", settlement="1", event="A"),
        _trade(end_ts=200, premium="0.50", settlement="0", event="B"),
    )
    # 100 - 0.40 + 1 = 100.60; then 100.60 - 0.50 + 0 = 100.10
    survived, cash = simulate_bankroll(trades, start=Decimal("100"))
    assert survived is True
    assert cash == Decimal("100.10")


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


def test_evaluate_trial_rejects_ruin_with_tiny_bankroll(tmp_path: Path) -> None:
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
        bankroll=Decimal("0.01"),
    )
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    value = evaluate_trial(trial, config=config, min_n=1)
    assert value == float("-inf")
    assert trial.user_attrs.get("reject") == "ruin"


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
        return evaluate_trial(trial, config=config, min_n=1)

    study.optimize(objective, n_trials=3, show_progress_bar=False)
    complete = [t for t in study.trials if t.value is not None and t.value > float("-inf")]
    assert complete
    assert complete[0].user_attrs["strategy"] == "s1"
    assert complete[0].user_attrs["minute"] == 55


def _seed_s1_db_with_exits(db_path: Path, *, n_events: int = 4) -> None:
    """Entry + post-entry quotes so stop/take can fire during tune trials."""
    connection = connect(db_path)
    initialize_schema(connection)
    for offset in range(n_events):
        close = RESEARCH_CLOSE.replace(hour=12 + offset)
        ticker = f"SYNTH-EVENT-TUNE-{offset:02d}"
        settled = SettledEvent(
            event=Event(
                event_ticker=ticker,
                close_ts=close,
                status=STATUS_COMPLETE,
                expiration_value=Decimal("150"),
            ),
            brackets=(
                Bracket(
                    ticker=f"{ticker}-BAND-A",
                    event_ticker=ticker,
                    floor_strike=Decimal("100"),
                    cap_strike=Decimal("199.99"),
                    won=True,
                ),
                Bracket(
                    ticker=f"{ticker}-BAND-B",
                    event_ticker=ticker,
                    floor_strike=Decimal("200"),
                    cap_strike=Decimal("299.99"),
                    won=False,
                ),
                Bracket(
                    ticker=f"{ticker}-BAND-C",
                    event_ticker=ticker,
                    floor_strike=Decimal("300"),
                    cap_strike=Decimal("399.99"),
                    won=False,
                ),
            ),
        )
        store_settled_events(connection, [settled])
        store_btc_candles(
            connection,
            [
                MinuteBar(
                    end_ts=checkpoint_end_ts(close, minute),
                    open="150",
                    high="150",
                    low="150",
                    close="150",
                )
                for minute in range(1, 56)
            ],
        )
        market = f"{ticker}-BAND-A"
        store_market_candles(
            connection,
            [
                MarketQuoteBar(
                    ticker=market,
                    end_ts=checkpoint_end_ts(close, 55),
                    yes_bid_close="0.50",
                    yes_ask_close="0.55",
                ),
                MarketQuoteBar(
                    ticker=market,
                    end_ts=checkpoint_end_ts(close, 56),
                    yes_bid_close="0.20",
                    yes_ask_close="0.22",
                ),
                MarketQuoteBar(
                    ticker=market,
                    end_ts=checkpoint_end_ts(close, 57),
                    yes_bid_close="0.70",
                    yes_ask_close="0.75",
                ),
            ],
        )
    connection.close()


def test_evaluate_trial_enqueued_stop_changes_pnl(tmp_path: Path) -> None:
    db_path = tmp_path / "bob.sqlite"
    _seed_s1_db_with_exits(db_path, n_events=4)
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
        bankroll=Decimal("100"),
    )
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(
        {
            "strategy": "s1",
            "minute": 55,
            "use_stop": True,
            "use_take": False,
            "stop_bid": 0.30,
            "stop_from": 55,
        }
    )

    def objective(trial: optuna.Trial) -> float:
        return evaluate_trial(trial, config=config, min_n=1)

    study.optimize(objective, n_trials=1, show_progress_bar=False)
    trial = study.trials[0]
    assert trial.value is not None and trial.value > float("-inf")
    assert Decimal(trial.user_attrs["stop_bid"]) == Decimal("0.3")
    assert trial.user_attrs["take_pct"] is None
    assert trial.user_attrs["stop_from"] == 55
    assert trial.user_attrs["stopped"] == 4
    assert trial.user_attrs["taken"] == 0
    assert Decimal(trial.user_attrs["gross"]) == Decimal("0.20") * 4 - Decimal(
        "0.55"
    ) * 4
    assert "final_cash" in trial.user_attrs


def test_evaluate_trial_enqueued_take_changes_pnl(tmp_path: Path) -> None:
    db_path = tmp_path / "bob.sqlite"
    _seed_s1_db_with_exits(db_path, n_events=4)
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
    study.enqueue_trial(
        {
            "strategy": "s1",
            "minute": 55,
            "use_stop": False,
            "use_take": True,
            "take_pct": 0.20,
            "stop_from": 55,
        }
    )

    def objective(trial: optuna.Trial) -> float:
        return evaluate_trial(trial, config=config, min_n=1)

    study.optimize(objective, n_trials=1, show_progress_bar=False)
    trial = study.trials[0]
    assert trial.user_attrs["taken"] == 4
    assert trial.user_attrs["stopped"] == 0
    assert Decimal(trial.user_attrs["take_pct"]) == Decimal("0.2")


def test_simulate_bankroll_orders_same_ts_by_event_ticker() -> None:
    # Same end_ts: ticker "A" before "B" — B would ruin if ordered first.
    trades = (
        _trade(end_ts=100, premium="0.60", settlement="0", event="B"),
        _trade(end_ts=100, premium="0.40", settlement="1", event="A"),
    )
    survived, cash = simulate_bankroll(trades, start=Decimal("1.00"))
    # A first: 1-0.40+1=1.60; then B: 1.60-0.60+0=1.00
    assert survived is True
    assert cash == Decimal("1.00")


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
            "--bankroll",
            "100",
        ],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert "events=4" in result.output
    assert "bankroll=100" in result.output
    assert "tune" in result.output.lower() or "top trials" in result.output
