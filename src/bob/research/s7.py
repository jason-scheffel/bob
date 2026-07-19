# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s7: dominant-bracket occupancy hold-to-settlement.

Buy YES/NO on the unique modal bracket among minutes 1..M when it
occupies at least half of those closes.

Treat this module as frozen. Prefer a new strategy module over editing s7.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from bob.browse import load_brackets, load_events, winning_bracket
from bob.research.common import (
    brackets_containing,
    finite_decimal,
    load_all_complete_events,
    load_minute_closes,
)

STRATEGY = "s7"
STRATEGY_SUMMARY = "dominant-bracket occupancy hold"

DEFAULT_CHECKPOINT_MINUTES = (45, 50, 55)
MIN_CHECKPOINT = 1

Side = Literal["yes", "no"]

ExclusionReason = Literal[
    "missing_bars",
    "bad_price",
    "no_bracket_match",
    "ambiguous_bracket",
    "bad_winner_invariant",
    "bad_expiration",
    "no_winner",
]

AbstentionReason = Literal[
    "no_unique_mode",
    "low_occupancy",
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
    minutes: tuple[MinuteStats, ...]


def _outcome_win(*, selected_won: bool, side: Side) -> bool:
    return selected_won if side == "yes" else not selected_won


def evaluate(
    connection: sqlite3.Connection,
    *,
    minutes: Sequence[int] = DEFAULT_CHECKPOINT_MINUTES,
    side: Side = "yes",
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    minute_list = tuple(minutes)
    if not minute_list:
        raise ValueError("minutes must be non-empty")
    if len(set(minute_list)) != len(minute_list):
        raise ValueError("minutes must be unique")
    for minute in minute_list:
        if not MIN_CHECKPOINT <= minute <= 59:
            raise ValueError(f"minute must be in {MIN_CHECKPOINT}..59, got {minute}")
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

    for event in events:
        brackets = load_brackets(connection, event.event_ticker)
        winner = winning_bracket(brackets)
        expiration = finite_decimal(event.expiration_value)
        by_ticker = {bracket.ticker: bracket for bracket in brackets}

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

            window = range(1, minute + 1)
            closes = load_minute_closes(connection, event.close_ts, window)
            if closes is None:
                exclusions[minute]["missing_bars"] += 1
                continue

            tickers: list[str] = []
            mapping_ok = True
            for close in closes:
                matches = brackets_containing(close, brackets)
                if len(matches) != 1:
                    exclusions[minute][
                        "no_bracket_match" if len(matches) == 0 else "ambiguous_bracket"
                    ] += 1
                    mapping_ok = False
                    break
                tickers.append(matches[0].ticker)
            if not mapping_ok:
                continue

            counts = Counter(tickers)
            top_count = counts.most_common(1)[0][1]
            leaders = [ticker for ticker, count in counts.items() if count == top_count]
            if len(leaders) != 1:
                abstentions[minute]["no_unique_mode"] += 1
                continue
            if top_count < math.ceil(minute / 2):
                abstentions[minute]["low_occupancy"] += 1
                continue
            selected = by_ticker[leaders[0]]
            if _outcome_win(selected_won=selected.won, side=side):
                wins[minute] += 1
            else:
                losses[minute] += 1

    return Report(
        strategy=STRATEGY,
        side=side,
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
    )
