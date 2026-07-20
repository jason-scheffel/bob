# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s23: aged unfinished sojourn-drawdown hold.

Buy YES/NO on the live closed bracket when contiguous unique-bracket
age is at least 15 minutes and unfinished sojourn drawdown is at least
10% of bracket width.

Treat this module as frozen. Prefer a new strategy module over editing s23.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from bob.browse import BracketRow, load_brackets, load_events, winning_bracket
from bob.research.common import (
    brackets_containing,
    checkpoint_end_ts,
    finite_decimal,
    load_all_complete_events,
    load_minute_closes,
)
from bob.research.trades import TradeObservation

STRATEGY = "s23"
STRATEGY_SUMMARY = "aged unfinished sojourn-drawdown hold"

DEFAULT_CHECKPOINT_MINUTES = (40, 45, 50, 55)
MIN_CHECKPOINT = 15
DEFAULT_AGE_MIN = 15
DEFAULT_DD_MIN = Decimal("0.10")

Side = Literal["yes", "no"]

ExclusionReason = Literal[
    "missing_bars",
    "no_bracket_match",
    "ambiguous_bracket",
    "bad_winner_invariant",
    "bad_expiration",
    "no_winner",
]

AbstentionReason = Literal[
    "open_ended_bracket",
    "young_age",
    "finished_pullback",
]


@dataclass(frozen=True, slots=True)
class MinuteStats:
    minute: int
    eligible: int
    wins: int
    losses: int
    abstentions: dict[str, int]
    exclusions: dict[str, int]

    @property
    def win_rate(self) -> float | None:
        if self.eligible == 0:
            return None
        return self.wins / self.eligible

    @property
    def abstained(self) -> int:
        return sum(self.abstentions.values())


@dataclass(frozen=True, slots=True)
class Report:
    strategy: str
    side: Side
    age_min: int
    dd_min: Decimal
    minutes: tuple[MinuteStats, ...]
    trades: tuple[TradeObservation, ...]


def contiguous_bracket_age(
    closes: Sequence[Decimal],
    brackets: Sequence[BracketRow],
    selected: BracketRow,
) -> int:
    """Bars in the contiguous unique-mapping suffix ending at the checkpoint."""
    age = 0
    for close in reversed(closes):
        matches = brackets_containing(close, brackets)
        if len(matches) != 1 or matches[0].ticker != selected.ticker:
            break
        age += 1
    return age


def sojourn_drawdown(
    closes: Sequence[Decimal],
    *,
    age: int,
    floor: Decimal,
    cap: Decimal,
) -> Decimal | None:
    """High-water unfinished drawdown over the sojourn, as a fraction of width."""
    if age < 1 or age > len(closes):
        return None
    width = cap - floor
    if width <= 0:
        return None
    sojourn = closes[-age:]
    high_water = max(sojourn)
    return (high_water - sojourn[-1]) / width


def _outcome_win(*, selected_won: bool, side: Side) -> bool:
    return selected_won if side == "yes" else not selected_won


def _classify(count: int) -> ExclusionReason | None:
    if count == 1:
        return None
    if count == 0:
        return "no_bracket_match"
    return "ambiguous_bracket"


def evaluate(
    connection: sqlite3.Connection,
    *,
    minutes: Sequence[int] = DEFAULT_CHECKPOINT_MINUTES,
    side: Side = "yes",
    age_min: int = DEFAULT_AGE_MIN,
    dd_min: Decimal = DEFAULT_DD_MIN,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    """Score aged unfinished sojourn-drawdown hold accuracy."""
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    if not isinstance(age_min, int) or age_min < 1:
        raise ValueError(f"age_min must be a positive int, got {age_min!r}")
    if not isinstance(dd_min, Decimal):
        dd_min = Decimal(str(dd_min))
    if not dd_min.is_finite() or dd_min < 0:
        raise ValueError(f"dd_min must be a finite non-negative Decimal, got {dd_min}")
    minute_list = tuple(minutes)
    if not minute_list:
        raise ValueError("minutes must be non-empty")
    if len(set(minute_list)) != len(minute_list):
        raise ValueError("minutes must be unique")
    for minute in minute_list:
        if not MIN_CHECKPOINT <= minute <= 59:
            raise ValueError(f"minute must be in {MIN_CHECKPOINT}..59, got {minute}")
        if minute < age_min:
            raise ValueError(
                f"minute {minute} is earlier than age_min={age_min}"
            )
    if (start is None) ^ (end is None):
        raise ValueError("start and end must both be set or both omitted")
    if start is not None and end is not None and start >= end:
        raise ValueError("start must be earlier than end")

    events = (
        load_all_complete_events(connection)
        if start is None
        else load_events(connection, start, end)
    )
    wins = Counter({minute: 0 for minute in minute_list})
    losses = Counter({minute: 0 for minute in minute_list})
    exclusions: dict[int, Counter[str]] = {minute: Counter() for minute in minute_list}
    abstentions: dict[int, Counter[str]] = {minute: Counter() for minute in minute_list}
    trades: list[TradeObservation] = []

    for event in events:
        brackets = load_brackets(connection, event.event_ticker)
        winner = winning_bracket(brackets)
        expiration = finite_decimal(event.expiration_value)

        for minute in minute_list:
            if winner is None:
                exclusions[minute]["no_winner"] += 1
                continue
            if expiration is None:
                exclusions[minute]["bad_expiration"] += 1
                continue
            settlement = brackets_containing(expiration, brackets)
            if len(settlement) != 1 or settlement[0] != winner:
                exclusions[minute]["bad_winner_invariant"] += 1
                continue

            closes = load_minute_closes(
                connection, event.close_ts, range(1, minute + 1)
            )
            if closes is None:
                exclusions[minute]["missing_bars"] += 1
                continue

            price = closes[-1]
            matches = brackets_containing(price, brackets)
            reason = _classify(len(matches))
            if reason is not None:
                exclusions[minute][reason] += 1
                continue
            selected = matches[0]
            if selected.floor_strike is None or selected.cap_strike is None:
                abstentions[minute]["open_ended_bracket"] += 1
                continue
            floor = finite_decimal(selected.floor_strike)
            cap = finite_decimal(selected.cap_strike)
            if floor is None or cap is None or floor >= cap:
                exclusions[minute]["no_bracket_match"] += 1
                continue

            age = contiguous_bracket_age(closes, brackets, selected)
            if age < age_min:
                abstentions[minute]["young_age"] += 1
                continue
            drawdown = sojourn_drawdown(closes, age=age, floor=floor, cap=cap)
            if drawdown is None or drawdown < dd_min:
                abstentions[minute]["finished_pullback"] += 1
                continue

            end_ts = checkpoint_end_ts(event.close_ts, minute)
            won = _outcome_win(selected_won=selected.won, side=side)
            if won:
                wins[minute] += 1
            else:
                losses[minute] += 1
            trades.append(
                TradeObservation(
                    event_ticker=event.event_ticker,
                    market_ticker=selected.ticker,
                    minute=minute,
                    end_ts=end_ts,
                    side=side,
                    won=won,
                )
            )

    return Report(
        strategy=STRATEGY,
        side=side,
        age_min=age_min,
        dd_min=dd_min,
        minutes=tuple(
            MinuteStats(
                minute=minute,
                eligible=wins[minute] + losses[minute],
                wins=wins[minute],
                losses=losses[minute],
                abstentions=dict(abstentions[minute]),
                exclusions=dict(exclusions[minute]),
            )
            for minute in minute_list
        ),
        trades=tuple(trades),
    )
