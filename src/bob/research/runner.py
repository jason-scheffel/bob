# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dispatch helpers for research strategies (win-rate and quote-sim P&L)."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from bob.db import connect, connect_readonly
from bob.research import (
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
)
from bob.research.pnl import QuoteSimReport, score_trades, score_trades_by_minute
from bob.research.s1 import Side
from bob.research.s12 import DEFAULT_TAU
from bob.research.s13 import DEFAULT_P_STAR
from bob.research.s14 import DEFAULT_Q_STAR
from bob.research.s15 import DEFAULT_DWELL, DEFAULT_MOVE

STRATEGY_MODULES = {
    "s1": s1,
    "s2": s2,
    "s3": s3,
    "s4": s4,
    "s5": s5,
    "s6": s6,
    "s7": s7,
    "s8": s8,
    "s9": s9,
    "s10": s10,
    "s11": s11,
    "s12": s12,
    "s13": s13,
    "s14": s14,
    "s15": s15,
}

STRATEGY_NAMES = tuple(STRATEGY_MODULES)


@dataclass(frozen=True, slots=True)
class StrategySummary:
    strategy: str
    summary: str
    side: Side
    eligible: int
    wins: int
    losses: int
    win_rate: float | None
    minutes: tuple[Any, ...]
    trades: int


@dataclass(frozen=True, slots=True)
class StrategyPnlSummary:
    strategy: str
    summary: str
    side: Side
    outcome: StrategySummary
    pnl: QuoteSimReport
    by_minute: tuple[tuple[int, QuoteSimReport], ...]


def evaluate_strategy(
    connection: sqlite3.Connection,
    name: str,
    *,
    minutes: Sequence[int],
    side: Side = "yes",
    start: datetime | None = None,
    end: datetime | None = None,
    tau: Decimal = DEFAULT_TAU,
    p_star: Decimal = DEFAULT_P_STAR,
    q_star: Decimal = DEFAULT_Q_STAR,
    move: Decimal = DEFAULT_MOVE,
    dwell: int = DEFAULT_DWELL,
) -> Any:
    module = STRATEGY_MODULES[name]
    kwargs: dict[str, Any] = {
        "minutes": minutes,
        "side": side,
        "start": start,
        "end": end,
    }
    if name == "s12":
        kwargs["tau"] = tau
    if name == "s13":
        kwargs["p_star"] = p_star
    if name == "s14":
        kwargs["p_star"] = p_star
        kwargs["q_star"] = q_star
    if name == "s15":
        kwargs["move"] = move
        kwargs["dwell"] = dwell
    return module.evaluate(connection, **kwargs)


def summarize_report(report: Any, *, summary: str) -> StrategySummary:
    eligible = sum(stats.eligible for stats in report.minutes)
    wins = sum(stats.wins for stats in report.minutes)
    losses = sum(stats.losses for stats in report.minutes)
    win_rate = None if eligible == 0 else wins / eligible
    return StrategySummary(
        strategy=report.strategy,
        summary=summary,
        side=report.side,
        eligible=eligible,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        minutes=tuple(report.minutes),
        trades=len(report.trades),
    )


def _open_db(db: Path | str, *, readonly: bool) -> sqlite3.Connection:
    if readonly:
        return connect_readonly(db)
    return connect(db)


def run_strategy_on_db(
    db: Path | str,
    name: str,
    *,
    minutes: Sequence[int],
    side: Side = "yes",
    start: datetime | None = None,
    end: datetime | None = None,
    tau: Decimal = DEFAULT_TAU,
    p_star: Decimal = DEFAULT_P_STAR,
    q_star: Decimal = DEFAULT_Q_STAR,
    move: Decimal = DEFAULT_MOVE,
    dwell: int = DEFAULT_DWELL,
    readonly: bool = False,
) -> StrategySummary:
    module = STRATEGY_MODULES[name]
    connection = _open_db(db, readonly=readonly)
    try:
        report = evaluate_strategy(
            connection,
            name,
            minutes=minutes,
            side=side,
            start=start,
            end=end,
            tau=tau,
            p_star=p_star,
            q_star=q_star,
            move=move,
            dwell=dwell,
        )
    finally:
        connection.close()
    return summarize_report(report, summary=module.STRATEGY_SUMMARY)


def run_strategy_pnl_on_db(
    db: Path | str,
    name: str,
    *,
    minutes: Sequence[int],
    side: Side = "yes",
    start: datetime | None = None,
    end: datetime | None = None,
    tau: Decimal = DEFAULT_TAU,
    p_star: Decimal = DEFAULT_P_STAR,
    q_star: Decimal = DEFAULT_Q_STAR,
    move: Decimal = DEFAULT_MOVE,
    dwell: int = DEFAULT_DWELL,
    readonly: bool = False,
) -> StrategyPnlSummary:
    module = STRATEGY_MODULES[name]
    connection = _open_db(db, readonly=readonly)
    try:
        report = evaluate_strategy(
            connection,
            name,
            minutes=minutes,
            side=side,
            start=start,
            end=end,
            tau=tau,
            p_star=p_star,
            q_star=q_star,
            move=move,
            dwell=dwell,
        )
        pnl = score_trades(connection, report.trades)
        by_minute = score_trades_by_minute(connection, report.trades, minutes)
    finally:
        connection.close()
    return StrategyPnlSummary(
        strategy=report.strategy,
        summary=module.STRATEGY_SUMMARY,
        side=report.side,
        outcome=summarize_report(report, summary=module.STRATEGY_SUMMARY),
        pnl=pnl,
        by_minute=by_minute,
    )


def run_all_strategies(
    db: Path | str,
    *,
    minutes: Sequence[int],
    side: Side = "yes",
    start: datetime | None = None,
    end: datetime | None = None,
    tau: Decimal = DEFAULT_TAU,
    p_star: Decimal = DEFAULT_P_STAR,
    q_star: Decimal = DEFAULT_Q_STAR,
    move: Decimal = DEFAULT_MOVE,
    dwell: int = DEFAULT_DWELL,
    workers: int | None = None,
    names: Sequence[str] = STRATEGY_NAMES,
) -> tuple[StrategySummary, ...]:
    """Evaluate strategies in a process pool; print order is ``names`` order."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    wanted = tuple(names)
    if workers is None:
        workers = min(len(wanted), os.cpu_count() or 1)
    workers = max(1, workers)
    if db == ":memory:":
        raise ValueError(
            "run_all_strategies requires an on-disk database "
            "(process workers cannot share :memory:)"
        )
    db_path = str(Path(db).resolve())

    payload = {
        "db": db_path,
        "minutes": tuple(minutes),
        "side": side,
        "start": start,
        "end": end,
        "tau": tau,
        "p_star": p_star,
        "q_star": q_star,
        "move": move,
        "dwell": dwell,
    }
    results: dict[str, StrategySummary] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker_strategy, name, payload): name for name in wanted
        }
        for future in as_completed(futures):
            summary = future.result()
            results[summary.strategy] = summary
    return tuple(results[name] for name in wanted)


def run_all_strategy_pnl(
    db: Path | str,
    *,
    minutes: Sequence[int],
    side: Side = "yes",
    start: datetime | None = None,
    end: datetime | None = None,
    tau: Decimal = DEFAULT_TAU,
    p_star: Decimal = DEFAULT_P_STAR,
    q_star: Decimal = DEFAULT_Q_STAR,
    move: Decimal = DEFAULT_MOVE,
    dwell: int = DEFAULT_DWELL,
    workers: int | None = None,
    names: Sequence[str] = STRATEGY_NAMES,
) -> tuple[StrategyPnlSummary, ...]:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    wanted = tuple(names)
    if workers is None:
        workers = min(len(wanted), os.cpu_count() or 1)
    workers = max(1, workers)
    if db == ":memory:":
        raise ValueError(
            "run_all_strategy_pnl requires an on-disk database "
            "(process workers cannot share :memory:)"
        )
    db_path = str(Path(db).resolve())

    payload = {
        "db": db_path,
        "minutes": tuple(minutes),
        "side": side,
        "start": start,
        "end": end,
        "tau": tau,
        "p_star": p_star,
        "q_star": q_star,
        "move": move,
        "dwell": dwell,
    }
    results: dict[str, StrategyPnlSummary] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker_pnl, name, payload): name for name in wanted
        }
        for future in as_completed(futures):
            summary = future.result()
            results[summary.strategy] = summary
    return tuple(results[name] for name in wanted)


def _worker_strategy(name: str, payload: dict[str, Any]) -> StrategySummary:
    return run_strategy_on_db(
        payload["db"],
        name,
        minutes=payload["minutes"],
        side=payload["side"],
        start=payload["start"],
        end=payload["end"],
        tau=payload["tau"],
        p_star=payload["p_star"],
        q_star=payload["q_star"],
        move=payload["move"],
        dwell=payload["dwell"],
        readonly=True,
    )


def _worker_pnl(name: str, payload: dict[str, Any]) -> StrategyPnlSummary:
    return run_strategy_pnl_on_db(
        payload["db"],
        name,
        minutes=payload["minutes"],
        side=payload["side"],
        start=payload["start"],
        end=payload["end"],
        tau=payload["tau"],
        p_star=payload["p_star"],
        q_star=payload["q_star"],
        move=payload["move"],
        dwell=payload["dwell"],
        readonly=True,
    )
