# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s2: stable-center current-bracket hold-to-settlement.

At a checkpoint minute, buy YES/NO on the live bracket only when the
recent path stayed inside that bracket and the checkpoint close is in
the middle 50% of a closed bracket. Otherwise abstain.

Treat this module as frozen. Prefer a new strategy module over editing s2.
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
    load_candle_ohlc,
    price_in_bracket,
)

STRATEGY = "s2"
STRATEGY_SUMMARY = "stable-center bracket hold"

DEFAULT_CHECKPOINT_MINUTES = (55,)
LOOKBACK_MINUTES = 5
EDGE_FRACTION = Decimal("0.25")

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

AbstentionReason = Literal[
    "open_ended_bracket",
    "unstable_path",
    "near_edge",
    "missing_lookback",
    "bad_lookback_price",
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


def lookback_minutes(checkpoint: int) -> tuple[int, ...]:
    """Minutes ending at ``checkpoint``, length ``LOOKBACK_MINUTES``."""
    if checkpoint < LOOKBACK_MINUTES:
        raise ValueError(
            f"checkpoint minute must be >= {LOOKBACK_MINUTES}, got {checkpoint}"
        )
    start = checkpoint - LOOKBACK_MINUTES + 1
    return tuple(range(start, checkpoint + 1))


def in_center_band(
    price: Decimal,
    floor_strike: str | None,
    cap_strike: str | None,
    *,
    edge_fraction: Decimal = EDGE_FRACTION,
) -> bool | None:
    """True if price is in the middle band; None if bracket is open-ended."""
    if floor_strike is None or cap_strike is None:
        return None
    floor = finite_decimal(floor_strike)
    cap = finite_decimal(cap_strike)
    if floor is None or cap is None:
        return None
    width = cap - floor
    if width <= 0:
        return None
    lo = floor + edge_fraction * width
    hi = cap - edge_fraction * width
    return lo <= price <= hi


def path_stays_in_bracket(
    bars: Sequence[tuple[Decimal, Decimal]],
    bracket: BracketRow,
) -> bool:
    """True when every high/low stays inside ``bracket``."""
    for high, low in bars:
        if not price_in_bracket(high, bracket.floor_strike, bracket.cap_strike):
            return False
        if not price_in_bracket(low, bracket.floor_strike, bracket.cap_strike):
            return False
    return True


def evaluate(
    connection: sqlite3.Connection,
    *,
    minutes: Sequence[int] = DEFAULT_CHECKPOINT_MINUTES,
    side: Side = "yes",
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    """Score stable-center current-bracket hold outcome accuracy."""
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    minute_list = tuple(minutes)
    if not minute_list:
        raise ValueError("minutes must be non-empty")
    if len(set(minute_list)) != len(minute_list):
        raise ValueError("minutes must be unique")
    for minute in minute_list:
        if not LOOKBACK_MINUTES <= minute <= 59:
            raise ValueError(f"minute must be in {LOOKBACK_MINUTES}..59, got {minute}")
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

            settlement_matches = brackets_containing(expiration, brackets)
            if len(settlement_matches) != 1 or settlement_matches[0] != winner:
                exclusions[minute]["bad_winner_invariant"] += 1
                continue

            window = lookback_minutes(minute)
            bars: list[tuple[Decimal, Decimal]] = []
            checkpoint_close: Decimal | None = None
            lookback_ok = True
            for window_minute in window:
                end_ts = checkpoint_end_ts(event.close_ts, window_minute)
                ohlc = load_candle_ohlc(connection, end_ts)
                if ohlc is None:
                    abstentions[minute]["missing_lookback"] += 1
                    lookback_ok = False
                    break
                high = finite_decimal(ohlc.high)
                low = finite_decimal(ohlc.low)
                close = finite_decimal(ohlc.close)
                if high is None or low is None or close is None:
                    abstentions[minute]["bad_lookback_price"] += 1
                    lookback_ok = False
                    break
                bars.append((high, low))
                if window_minute == minute:
                    checkpoint_close = close
            if not lookback_ok:
                continue
            if checkpoint_close is None:
                exclusions[minute]["missing_bar"] += 1
                continue

            matches = brackets_containing(checkpoint_close, brackets)
            reason = _classify_match_count(len(matches))
            if reason is not None:
                exclusions[minute][reason] += 1
                continue
            selected = matches[0]

            center = in_center_band(
                checkpoint_close,
                selected.floor_strike,
                selected.cap_strike,
            )
            if center is None:
                abstentions[minute]["open_ended_bracket"] += 1
                continue
            if not path_stays_in_bracket(bars, selected):
                abstentions[minute]["unstable_path"] += 1
                continue
            if not center:
                abstentions[minute]["near_edge"] += 1
                continue

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
                abstentions=dict(abstentions[minute]),
                exclusions=dict(exclusions[minute]),
            )
        )
    return Report(
        strategy=STRATEGY,
        side=side,
        minutes=tuple(stats),
    )
