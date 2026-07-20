# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s22: asymmetric dual-buffer residual-range hold.

Buy YES/NO on the live closed bracket when the nearer wall clears a
quarter projected RMS move and the farther wall clears three-quarters
(d_near ≥ 0.25 σ√R and d_far ≥ 0.75 σ√R).

Treat this module as frozen. Prefer a new strategy module over editing s22.
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
    finite_decimal,
    load_all_complete_events,
    load_minute_closes,
)
from bob.research.s21 import rms_close_diff_sigma
from bob.research.trades import TradeObservation

STRATEGY = "s22"
STRATEGY_SUMMARY = "asymmetric dual-buffer residual-range hold"

DEFAULT_CHECKPOINT_MINUTES = (40, 45, 50, 55)
MIN_CHECKPOINT = 2
DEFAULT_NEAR_MULT = Decimal("0.25")
DEFAULT_FAR_MULT = Decimal("0.75")

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
    "asymmetric_weak",
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
    near_mult: Decimal
    far_mult: Decimal
    minutes: tuple[MinuteStats, ...]
    trades: tuple[TradeObservation, ...]


def wall_distances(
    price: Decimal,
    floor_strike: str | None,
    cap_strike: str | None,
) -> tuple[Decimal, Decimal] | None:
    """Return (d_near, d_far) for a closed bracket, or None if open-ended."""
    if floor_strike is None or cap_strike is None:
        return None
    floor = finite_decimal(floor_strike)
    cap = finite_decimal(cap_strike)
    if floor is None or cap is None or floor >= cap:
        return None
    if not (floor <= price <= cap):
        return None
    below = price - floor
    above = cap - price
    return min(below, above), max(below, above)


def asymmetric_buffers_clear(
    *,
    d_near: Decimal,
    d_far: Decimal,
    sigma: Decimal,
    remaining: int,
    near_mult: Decimal,
    far_mult: Decimal,
) -> bool:
    """True when dual asymmetric residual buffers clear (or σ = 0)."""
    if remaining <= 0:
        return False
    if sigma == 0:
        return True
    scale = sigma * Decimal(str(math.sqrt(remaining)))
    return d_near >= near_mult * scale and d_far >= far_mult * scale


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
    near_mult: Decimal = DEFAULT_NEAR_MULT,
    far_mult: Decimal = DEFAULT_FAR_MULT,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    """Score asymmetric dual-buffer residual-range hold accuracy."""
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    if not isinstance(near_mult, Decimal):
        near_mult = Decimal(str(near_mult))
    if not isinstance(far_mult, Decimal):
        far_mult = Decimal(str(far_mult))
    if not near_mult.is_finite() or near_mult < 0:
        raise ValueError(f"near_mult must be a finite non-negative Decimal, got {near_mult}")
    if not far_mult.is_finite() or far_mult < 0:
        raise ValueError(f"far_mult must be a finite non-negative Decimal, got {far_mult}")
    if near_mult > far_mult:
        raise ValueError("near_mult must be ≤ far_mult")
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
            distances = wall_distances(
                price, selected.floor_strike, selected.cap_strike
            )
            if distances is None:
                abstentions[minute]["open_ended_bracket"] += 1
                continue
            d_near, d_far = distances

            sigma = rms_close_diff_sigma(closes)
            remaining = 60 - minute
            if not asymmetric_buffers_clear(
                d_near=d_near,
                d_far=d_far,
                sigma=sigma,
                remaining=remaining,
                near_mult=near_mult,
                far_mult=far_mult,
            ):
                abstentions[minute]["asymmetric_weak"] += 1
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
        near_mult=near_mult,
        far_mult=far_mult,
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
