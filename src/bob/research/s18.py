# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s18: matched-horizon edgeward current-bracket hold.

Buy YES/NO on the live closed bracket when price occupied that bracket
one remaining-time horizon earlier and has since moved strictly farther
from its midpoint. The edgeward move seeks a quote discount without giving
up s8's bracket-persistence confirmation.

Treat this module as frozen. Prefer a new strategy module over editing s18.
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
    finite_decimal,
    load_all_complete_events,
    load_minute_closes,
)
from bob.research.s8 import confirmation_minute
from bob.research.trades import TradeObservation

STRATEGY = "s18"
STRATEGY_SUMMARY = "matched-horizon edgeward current bracket"

DEFAULT_CHECKPOINT_MINUTES = (40, 45, 50, 55)
MIN_CHECKPOINT = 31

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
    "unconfirmed_bracket",
    "not_edgeward",
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
    trades: tuple[TradeObservation, ...]


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
    """Score matched-horizon edgeward current-bracket hold accuracy."""
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
                connection,
                event.close_ts,
                (confirmation_minute(minute), minute),
            )
            if closes is None:
                exclusions[minute]["missing_bars"] += 1
                continue
            earlier_matches = brackets_containing(closes[0], brackets)
            current_matches = brackets_containing(closes[1], brackets)
            reason = _classify(len(earlier_matches))
            if reason is None:
                reason = _classify(len(current_matches))
            if reason is not None:
                exclusions[minute][reason] += 1
                continue

            selected = current_matches[0]
            if selected.floor_strike is None or selected.cap_strike is None:
                abstentions[minute]["open_ended_bracket"] += 1
                continue
            floor = finite_decimal(selected.floor_strike)
            cap = finite_decimal(selected.cap_strike)
            if floor is None or cap is None or floor >= cap:
                exclusions[minute]["bad_price"] += 1
                continue
            if earlier_matches[0] != selected:
                abstentions[minute]["unconfirmed_bracket"] += 1
                continue
            midpoint = (floor + cap) / 2
            if abs(closes[1] - midpoint) <= abs(closes[0] - midpoint):
                abstentions[minute]["not_edgeward"] += 1
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
