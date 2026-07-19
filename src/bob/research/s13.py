# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s13: regime-pooled terminal-probability current-bracket hold.

Estimate per-minute Parkinson variance from the current hour and the
prior hour, form a Gaussian terminal scale over the remaining minutes,
and buy YES/NO on the live closed bracket only when the implied win
probability clears ``p_star``.

Treat this module as frozen. Prefer a new strategy module over editing s13.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

from bob.browse import load_brackets, load_events, winning_bracket
from bob.research.common import (
    brackets_containing,
    checkpoint_end_ts,
    finite_decimal,
    load_all_complete_events,
    load_candle_ohlc,
)

STRATEGY = "s13"
STRATEGY_SUMMARY = "regime-pooled terminal-probability current bracket"

DEFAULT_CHECKPOINT_MINUTES = (45, 50, 55)
MIN_CHECKPOINT = 30
DEFAULT_P_STAR = Decimal("0.70")
PARKINSON_DENOM = 4 * math.log(2)
MIN_PRIOR_BARS = 45

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
    "low_terminal_prob",
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
    p_star: Decimal
    minutes: tuple[MinuteStats, ...]


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def load_high_low(
    connection: sqlite3.Connection,
    close_ts: datetime,
    minutes: Sequence[int],
) -> list[tuple[Decimal, Decimal]] | None:
    """Load (high, low) pairs; None if any bar is missing/invalid."""
    bars: list[tuple[Decimal, Decimal]] = []
    for minute in minutes:
        bar = load_candle_ohlc(connection, checkpoint_end_ts(close_ts, minute))
        if bar is None:
            return None
        high = finite_decimal(bar.high)
        low = finite_decimal(bar.low)
        if high is None or low is None:
            return None
        bars.append((high, low))
    return bars


def mean_parkinson_variance(bars: Sequence[tuple[Decimal, Decimal]]) -> Decimal:
    """Mean per-minute Parkinson variance in dollar² units."""
    if not bars:
        raise ValueError("bars must be non-empty")
    total = sum((high - low) ** 2 for high, low in bars)
    return total / (Decimal(str(PARKINSON_DENOM)) * Decimal(len(bars)))


def terminal_win_probability(
    *,
    price: Decimal,
    floor: Decimal,
    cap: Decimal,
    sigma: Decimal,
    remaining: int,
) -> Decimal:
    """Gaussian direction-blind P(print stays in [floor, cap])."""
    if remaining <= 0:
        raise ValueError("remaining must be positive")
    if sigma <= 0:
        return Decimal("1") if floor < price < cap else Decimal("0")
    scale = float(sigma) * math.sqrt(remaining)
    above = float(cap - price) / scale
    below = float(price - floor) / scale
    return Decimal(str(_phi(above) - _phi(-below)))


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
    p_star: Decimal = DEFAULT_P_STAR,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    """Score regime-pooled terminal-probability hold accuracy."""
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    if not isinstance(p_star, Decimal):
        p_star = Decimal(str(p_star))
    if not p_star.is_finite() or not (Decimal("0") < p_star < Decimal("1")):
        raise ValueError(f"p_star must be in (0, 1), got {p_star}")
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
        prior_close = event.close_ts - timedelta(hours=1)
        prior_bars = load_high_low(connection, prior_close, range(1, 60))
        use_prior = prior_bars is not None and len(prior_bars) >= MIN_PRIOR_BARS

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

            current_bars = load_high_low(
                connection, event.close_ts, range(1, minute + 1)
            )
            if current_bars is None:
                exclusions[minute]["missing_bars"] += 1
                continue
            close_bar = load_candle_ohlc(
                connection, checkpoint_end_ts(event.close_ts, minute)
            )
            if close_bar is None:
                exclusions[minute]["missing_bars"] += 1
                continue
            price = finite_decimal(close_bar.close)
            if price is None:
                exclusions[minute]["bad_price"] += 1
                continue

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
            if floor is None or cap is None:
                exclusions[minute]["bad_price"] += 1
                continue

            if use_prior:
                assert prior_bars is not None
                range_sq = sum((high - low) ** 2 for high, low in prior_bars) + sum(
                    (high - low) ** 2 for high, low in current_bars
                )
                count = len(prior_bars) + len(current_bars)
                variance = range_sq / (Decimal(str(PARKINSON_DENOM)) * Decimal(count))
            else:
                variance = mean_parkinson_variance(current_bars)
            sigma = variance.sqrt()
            remaining = 60 - minute
            p_hat = terminal_win_probability(
                price=price,
                floor=floor,
                cap=cap,
                sigma=sigma,
                remaining=remaining,
            )
            if p_hat < p_star:
                abstentions[minute]["low_terminal_prob"] += 1
                continue

            if _outcome_win(selected_won=selected.won, side=side):
                wins[minute] += 1
            else:
                losses[minute] += 1

    return Report(
        strategy=STRATEGY,
        side=side,
        p_star=p_star,
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
