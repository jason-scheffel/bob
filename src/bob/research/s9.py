# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s9: matched-horizon excursion-buffered current-bracket hold.

Buy YES/NO on the live closed bracket only when its nearest edge is at
least as far away as the largest high/low excursion observed over a
trailing window equal to the time remaining before settlement.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from bob.browse import load_brackets, load_events, winning_bracket
from bob.research.common import (
    brackets_containing,
    checkpoint_end_ts,
    distance_to_nearest_edge,
    finite_decimal,
    load_all_complete_events,
    load_candle_ohlc,
)

STRATEGY = "s9"
STRATEGY_SUMMARY = "matched-horizon excursion-buffered current bracket"

DEFAULT_CHECKPOINT_MINUTES = (45, 50, 55)
MIN_CHECKPOINT = 30

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
    "open_ended_bracket",
    "thin_excursion_buffer",
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


def horizon_minutes(checkpoint: int) -> range:
    """Trailing minutes with length equal to minutes until settlement."""
    if not MIN_CHECKPOINT <= checkpoint <= 59:
        raise ValueError(
            f"checkpoint minute must be in {MIN_CHECKPOINT}..59, got {checkpoint}"
        )
    remaining = 60 - checkpoint
    return range(checkpoint - remaining + 1, checkpoint + 1)


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
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    """Score matched-horizon excursion-buffered hold accuracy."""
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

            highs: list[Decimal] = []
            lows: list[Decimal] = []
            price: Decimal | None = None
            valid = True
            for window_minute in horizon_minutes(minute):
                bar = load_candle_ohlc(
                    connection, checkpoint_end_ts(event.close_ts, window_minute)
                )
                if bar is None:
                    exclusions[minute]["missing_bars"] += 1
                    valid = False
                    break
                high = finite_decimal(bar.high)
                low = finite_decimal(bar.low)
                close = finite_decimal(bar.close)
                if high is None or low is None or close is None:
                    exclusions[minute]["bad_price"] += 1
                    valid = False
                    break
                highs.append(high)
                lows.append(low)
                if window_minute == minute:
                    price = close
            if not valid:
                continue
            assert price is not None

            matches = brackets_containing(price, brackets)
            reason = _classify(len(matches))
            if reason is not None:
                exclusions[minute][reason] += 1
                continue
            selected = matches[0]
            distance = distance_to_nearest_edge(
                price, selected.floor_strike, selected.cap_strike
            )
            if distance is None:
                abstentions[minute]["open_ended_bracket"] += 1
                continue
            excursion = max(max(highs) - price, price - min(lows))
            if distance < excursion:
                abstentions[minute]["thin_excursion_buffer"] += 1
                continue

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
