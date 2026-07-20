# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Optuna HPO over checkpoint minute + stop/take overlays (frozen decide rules)."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from bob.browse import load_events
from bob.db import connect_readonly
from bob.research.common import load_all_complete_events
from bob.research.pnl import QuoteSimTrade
from bob.research.runner import STRATEGY_MODULES, run_strategy_pnl_on_db
from bob.research.s1 import Side

DEFAULT_STRATEGIES = ("s20", "s21", "s22", "s23")
DEFAULT_MINUTES = (40, 45, 50, 55)
DEFAULT_MIN_FRAC = Decimal("0.25")
DEFAULT_TRIALS = 200
DEFAULT_TOP_K = 5
DEFAULT_BANKROLL = Decimal("100")

_REJECT = float("-inf")
_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class TuneConfig:
    db: Path
    strategies: tuple[str, ...]
    minutes: tuple[int, ...]
    side: Side
    start: datetime | None
    end: datetime | None
    min_frac: Decimal
    n_trials: int
    n_jobs: int
    bankroll: Decimal = DEFAULT_BANKROLL
    top_k: int = DEFAULT_TOP_K


@dataclass(frozen=True, slots=True)
class TuneResult:
    study: optuna.Study
    n_events: int
    min_n: int
    config: TuneConfig


def parse_strategy_names(value: str) -> tuple[str, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("strategies must be a non-empty comma list")
    names: list[str] = []
    for part in parts:
        if part not in STRATEGY_MODULES:
            known = ", ".join(STRATEGY_MODULES)
            raise ValueError(f"unknown strategy {part!r}; known: {known}")
        names.append(part)
    if len(set(names)) != len(names):
        raise ValueError("strategies must be unique")
    return tuple(names)


def count_complete_events(
    db: Path | str,
    *,
    start: datetime | None,
    end: datetime | None,
) -> int:
    if (start is None) ^ (end is None):
        raise ValueError("start and end must both be set or both omitted")
    if start is not None and end is not None and start >= end:
        raise ValueError("start must be earlier than end")
    connection = connect_readonly(db)
    try:
        if start is None:
            events = load_all_complete_events(connection)
        else:
            assert end is not None
            events = load_events(connection, start, end)
    finally:
        connection.close()
    return len(events)


def min_eligible_trades(*, n_events: int, min_frac: Decimal) -> int:
    if n_events < 1:
        raise ValueError("n_events must be >= 1")
    if not isinstance(min_frac, Decimal):
        min_frac = Decimal(str(min_frac))
    if not min_frac.is_finite() or not (Decimal("0") < min_frac <= Decimal("1")):
        raise ValueError(f"min_frac must be in (0, 1], got {min_frac}")
    return int(
        (min_frac * Decimal(n_events)).to_integral_value(rounding=ROUND_CEILING)
    )


def simulate_bankroll(
    trades: Sequence[QuoteSimTrade],
    *,
    start: Decimal,
) -> tuple[bool, Decimal]:
    """Walk 1-contract trades in time; return (survived, final_cash)."""
    if not isinstance(start, Decimal):
        start = Decimal(str(start))
    if not start.is_finite() or start <= _ZERO:
        raise ValueError(f"bankroll start must be a positive Decimal, got {start}")
    cash = start
    ordered = sorted(
        trades,
        key=lambda t: (t.observation.end_ts, t.observation.event_ticker),
    )
    for trade in ordered:
        if cash < trade.premium:
            return False, cash
        cash -= trade.premium
        cash += trade.settlement
        if cash <= _ZERO:
            return False, cash
    return True, cash


def _suggest_overlays(trial: optuna.Trial) -> tuple[Decimal | None, Decimal | None, int]:
    use_stop = trial.suggest_categorical("use_stop", [False, True])
    use_take = trial.suggest_categorical("use_take", [False, True])
    stop_bid: Decimal | None = None
    take_pct: Decimal | None = None
    if use_stop:
        stop_bid = Decimal(
            str(trial.suggest_float("stop_bid", 0.05, 0.45, step=0.01))
        )
    if use_take:
        take_pct = Decimal(
            str(trial.suggest_float("take_pct", 0.05, 0.50, step=0.01))
        )
    if use_stop or use_take:
        stop_from = trial.suggest_int("stop_from", 50, 58)
    else:
        stop_from = 55
    return stop_bid, take_pct, stop_from


def evaluate_trial(
    trial: optuna.Trial,
    *,
    config: TuneConfig,
    min_n: int,
) -> float:
    strategy = trial.suggest_categorical("strategy", list(config.strategies))
    minute = trial.suggest_categorical("minute", list(config.minutes))
    stop_bid, take_pct, stop_from = _suggest_overlays(trial)

    summary = run_strategy_pnl_on_db(
        config.db,
        strategy,
        minutes=(minute,),
        side=config.side,
        start=config.start,
        end=config.end,
        stop_bid=stop_bid,
        take_pct=take_pct,
        stop_from=stop_from,
        readonly=True,
    )
    pnl = summary.pnl
    trial.set_user_attr("strategy", strategy)
    trial.set_user_attr("minute", minute)
    trial.set_user_attr("stop_bid", None if stop_bid is None else str(stop_bid))
    trial.set_user_attr("take_pct", None if take_pct is None else str(take_pct))
    trial.set_user_attr("stop_from", stop_from)
    trial.set_user_attr("n", pnl.quote_eligible)
    trial.set_user_attr("gross", str(pnl.gross))
    trial.set_user_attr("premium", str(pnl.premium))
    trial.set_user_attr("stopped", pnl.stopped)
    trial.set_user_attr("taken", pnl.taken)

    if pnl.quote_eligible < min_n:
        trial.set_user_attr("reject", "n_floor")
        return _REJECT
    if pnl.premium <= 0:
        trial.set_user_attr("reject", "zero_premium")
        return _REJECT
    survived, final_cash = simulate_bankroll(pnl.trades, start=config.bankroll)
    trial.set_user_attr("bankroll", str(config.bankroll))
    trial.set_user_attr("final_cash", str(final_cash))
    if not survived:
        trial.set_user_attr("reject", "ruin")
        return _REJECT
    ret = pnl.return_on_premium
    if ret is None:
        trial.set_user_attr("reject", "no_return")
        return _REJECT
    value = float(ret)
    trial.set_user_attr("return", value)
    return value


class _RichProgressCallback:
    def __init__(self, *, n_trials: int, console: Console) -> None:
        self._n_trials = n_trials
        self._console = console
        self._progress = Progress(
            TextColumn("[bold]tune[/bold]"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[best]}"),
            console=console,
            transient=False,
        )
        self._task_id: Any | None = None
        self._started = False

    def start(self) -> None:
        self._progress.start()
        self._task_id = self._progress.add_task(
            "tune", total=self._n_trials, best="best=—"
        )
        self._started = True

    def stop(self) -> None:
        if self._started:
            self._progress.stop()
            self._started = False

    def __call__(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if self._task_id is None:
            return
        best = "best=—"
        try:
            if math.isfinite(study.best_value):
                best = f"best={study.best_value * 100:.1f}%"
        except ValueError:
            pass
        self._progress.update(self._task_id, advance=1, best=best)


def run_tune(config: TuneConfig, *, console: Console | None = None) -> TuneResult:
    if not config.strategies:
        raise ValueError("strategies must be non-empty")
    if not config.minutes:
        raise ValueError("minutes must be non-empty")
    if config.n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if config.n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    if not isinstance(config.bankroll, Decimal):
        raise ValueError("bankroll must be a Decimal")
    if not config.bankroll.is_finite() or config.bankroll <= _ZERO:
        raise ValueError(f"bankroll must be positive, got {config.bankroll}")
    for name in config.strategies:
        if name not in STRATEGY_MODULES:
            raise ValueError(f"unknown strategy {name!r}")

    n_events = count_complete_events(
        config.db, start=config.start, end=config.end
    )
    if n_events < 1:
        raise ValueError("no complete events in the selected window")
    min_n = min_eligible_trades(n_events=n_events, min_frac=config.min_frac)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(),
    )
    out = console if console is not None else Console(stderr=True)
    callback = _RichProgressCallback(n_trials=config.n_trials, console=out)

    def objective(trial: optuna.Trial) -> float:
        return evaluate_trial(trial, config=config, min_n=min_n)

    callback.start()
    try:
        study.optimize(
            objective,
            n_trials=config.n_trials,
            n_jobs=config.n_jobs,
            show_progress_bar=False,
            callbacks=[callback],
        )
    finally:
        callback.stop()

    return TuneResult(
        study=study, n_events=n_events, min_n=min_n, config=config
    )


def _trial_rows(study: optuna.Study, *, top_k: int) -> list[optuna.trial.FrozenTrial]:
    finished = [
        t
        for t in study.get_trials(deepcopy=False)
        if t.state == TrialState.COMPLETE and t.value is not None and math.isfinite(t.value)
    ]
    finished.sort(key=lambda t: t.value or _REJECT, reverse=True)
    return finished[:top_k]


def print_tune_results(result: TuneResult, *, console: Console) -> None:
    cfg = result.config
    console.print(
        f"[dim]events={result.n_events} · min n="
        f"{result.min_n} ({cfg.min_frac}×events) · "
        f"bankroll={cfg.bankroll} · trials={cfg.n_trials}[/dim]"
    )
    rows = _trial_rows(result.study, top_k=cfg.top_k)
    if not rows:
        console.print(
            "[yellow]No trials met the n-floor / ruin / return constraints.[/yellow]"
        )
        return

    table = Table(title="research tune · top trials")
    table.add_column("#", justify="right")
    table.add_column("strategy")
    table.add_column("M", justify="right")
    table.add_column("stop_bid", justify="right")
    table.add_column("take_pct", justify="right")
    table.add_column("stop_from", justify="right")
    table.add_column("n", justify="right")
    table.add_column("gross", justify="right")
    table.add_column("return", justify="right")
    table.add_column("final_cash", justify="right")
    for rank, trial in enumerate(rows, start=1):
        attrs = trial.user_attrs
        ret = attrs.get("return")
        ret_s = "—" if ret is None else f"{float(ret) * 100:.1f}%"
        table.add_row(
            str(rank),
            str(attrs.get("strategy", "—")),
            str(attrs.get("minute", "—")),
            str(attrs.get("stop_bid") or "—"),
            str(attrs.get("take_pct") or "—"),
            str(attrs.get("stop_from", "—")),
            str(attrs.get("n", "—")),
            str(attrs.get("gross", "—")),
            ret_s,
            str(attrs.get("final_cash", "—")),
        )
    console.print(table)
