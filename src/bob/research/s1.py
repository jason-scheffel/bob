# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s1: current-bracket checkpoint hold-to-settlement outcome accuracy.

Treat this module as frozen. Prefer a new strategy module over editing s1.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from bob.browse import load_brackets, load_events, winning_bracket
from bob.research.common import (
    brackets_containing,
    checkpoint_end_ts,
    finite_decimal,
    load_all_complete_events,
    load_candle_close,
)

STRATEGY = "s1"
STRATEGY_SUMMARY = "current-bracket checkpoint hold"

DEFAULT_CHECKPOINT_MINUTES = (40, 45, 50, 55)

Side = Literal["yes", "no"]

ExclusionReason = Literal[
    "missing_bar",
    "bad_price",
    "no_bracket_match",
    "ambiguous_bracket",
    "bad_winner_invariant",
    "bad_expiration",
    "no_winner",
]


@dataclass(frozen=True, slots=True)
class MinuteStats:
    minute: int
    eligible: int
    wins: int
    losses: int
    exclusions: dict[str, int]

    @property
    def win_rate(self) -> float | None:
        if self.eligible == 0:
            return None
        return self.wins / self.eligible


@dataclass(frozen=True, slots=True)
class Report:
    strategy: str
    side: Side
    minutes: tuple[MinuteStats, ...]


def _classify_match_count(count: int) -> ExclusionReason | None:
    if count == 1:
        return None
    if count == 0:
        return "no_bracket_match"
    return "ambiguous_bracket"


def _outcome_win(*, selected_won: bool, side: Side) -> bool:
    if side == "yes":
        return selected_won
    return not selected_won


def evaluate(
    connection: sqlite3.Connection,
    *,
    minutes: Sequence[int] = DEFAULT_CHECKPOINT_MINUTES,
    side: Side = "yes",
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    """Score current-bracket buy-and-hold outcome accuracy by checkpoint."""
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    minute_list = tuple(minutes)
    if not minute_list:
        raise ValueError("minutes must be non-empty")
    if len(set(minute_list)) != len(minute_list):
        raise ValueError("minutes must be unique")
    for minute in minute_list:
        if not 1 <= minute <= 59:
            raise ValueError(f"minute must be in 1..59, got {minute}")
    if (start is None) ^ (end is None):
        raise ValueError("start and end must both be set or both omitted")
    if start is not None and end is not None and start >= end:
        raise ValueError("start must be earlier than end")

    if start is None:
        events = load_all_complete_events(connection)
    else:
        events = load_events(connection, start, end)

    wins = Counter({minute: 0 for minute in minute_list})
    losses = Counter({minute: 0 for minute in minute_list})
    exclusions: dict[int, Counter[str]] = {minute: Counter() for minute in minute_list}

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

            settlement_matches = brackets_containing(expiration, brackets)
            if len(settlement_matches) != 1 or settlement_matches[0] != winner:
                exclusions[minute]["bad_winner_invariant"] += 1
                continue

            end_ts = checkpoint_end_ts(event.close_ts, minute)
            close = load_candle_close(connection, end_ts)
            if close is None:
                exclusions[minute]["missing_bar"] += 1
                continue
            price = finite_decimal(close)
            if price is None:
                exclusions[minute]["bad_price"] += 1
                continue

            matches = brackets_containing(price, brackets)
            reason = _classify_match_count(len(matches))
            if reason is not None:
                exclusions[minute][reason] += 1
                continue

            selected = matches[0]
            if _outcome_win(selected_won=selected.won, side=side):
                wins[minute] += 1
            else:
                losses[minute] += 1

    stats = []
    for minute in minute_list:
        eligible = wins[minute] + losses[minute]
        stats.append(
            MinuteStats(
                minute=minute,
                eligible=eligible,
                wins=wins[minute],
                losses=losses[minute],
                exclusions=dict(exclusions[minute]),
            )
        )
    return Report(
        strategy=STRATEGY,
        side=side,
        minutes=tuple(stats),
    )
