# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s21: residual-range lock current-bracket hold.

Buy YES/NO on the live closed bracket when nearest-edge distance clears
half a projected RMS one-minute move over remaining time
(z = d_edge / (σ √R) ≥ 0.5).

Treat this module as frozen. Prefer a new strategy module over editing s21.
"""

from __future__ import annotations

import math
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
    load_minute_closes,
)
from bob.research.trades import TradeObservation

STRATEGY = "s21"
STRATEGY_SUMMARY = "residual-range lock current bracket"

DEFAULT_CHECKPOINT_MINUTES = (40, 45, 50, 55)
MIN_CHECKPOINT = 2
DEFAULT_Z_STAR = Decimal("0.50")

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
    "weak_lock",
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
    z_star: Decimal
    minutes: tuple[MinuteStats, ...]
    trades: tuple[TradeObservation, ...]


def rms_close_diff_sigma(closes: Sequence[Decimal]) -> Decimal:
    """RMS of successive close diffs; 0 when all closes are identical."""
    if len(closes) < 2:
        raise ValueError("closes must contain at least two prices")
    total = sum((closes[i] - closes[i - 1]) ** 2 for i in range(1, len(closes)))
    return (total / Decimal(len(closes) - 1)).sqrt()


def residual_lock_z(
    *,
    edge_distance: Decimal,
    sigma: Decimal,
    remaining: int,
) -> Decimal | None:
    """Return lock z-score, or None when remaining time is non-positive."""
    if remaining <= 0:
        return None
    if sigma == 0:
        return None  # caller treats as +∞ / automatic pass
    return edge_distance / (sigma * Decimal(str(math.sqrt(remaining))))


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
    z_star: Decimal = DEFAULT_Z_STAR,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    """Score residual-range lock hold accuracy."""
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    if not isinstance(z_star, Decimal):
        z_star = Decimal(str(z_star))
    if not z_star.is_finite() or z_star < 0:
        raise ValueError(f"z_star must be a finite non-negative Decimal, got {z_star}")
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
            edge = distance_to_nearest_edge(
                price, selected.floor_strike, selected.cap_strike
            )
            if edge is None:
                abstentions[minute]["open_ended_bracket"] += 1
                continue

            sigma = rms_close_diff_sigma(closes)
            remaining = 60 - minute
            if sigma == 0:
                locked = True
            else:
                z = residual_lock_z(
                    edge_distance=edge, sigma=sigma, remaining=remaining
                )
                locked = z is not None and z >= z_star
            if not locked:
                abstentions[minute]["weak_lock"] += 1
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
        z_star=z_star,
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
