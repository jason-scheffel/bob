# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s14: two-clock vol disagreement current-bracket hold.

Compare a matched-horizon Parkinson clock (trailing R bars) with a
whole-hour clock (bars 1..M). Trade YES/NO on the live closed bracket
only when the recent clock clears ``p_star`` and the whole-hour clock
stays at or below ``q_star`` ("calm after the storm").

Treat this module as frozen. Prefer a new strategy module over editing s14.
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
    load_candle_ohlc,
)
from bob.research.s13 import load_high_low, mean_parkinson_variance, terminal_win_probability
from bob.research.trades import TradeObservation

STRATEGY = "s14"
STRATEGY_SUMMARY = "two-clock vol-disagreement current bracket"

DEFAULT_CHECKPOINT_MINUTES = (40, 45, 50, 55)
MIN_CHECKPOINT = 35
DEFAULT_P_STAR = Decimal("0.70")
DEFAULT_Q_STAR = Decimal("0.60")

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
    "low_quality",
    "already_priced",
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
    q_star: Decimal
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
    p_star: Decimal = DEFAULT_P_STAR,
    q_star: Decimal = DEFAULT_Q_STAR,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    """Score two-clock vol-disagreement hold accuracy."""
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    if not isinstance(p_star, Decimal):
        p_star = Decimal(str(p_star))
    if not isinstance(q_star, Decimal):
        q_star = Decimal(str(q_star))
    if not p_star.is_finite() or not (Decimal("0") < p_star < Decimal("1")):
        raise ValueError(f"p_star must be in (0, 1), got {p_star}")
    if not q_star.is_finite() or not (Decimal("0") < q_star < Decimal("1")):
        raise ValueError(f"q_star must be in (0, 1), got {q_star}")
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

            remaining = 60 - minute
            current_bars = load_high_low(
                connection, event.close_ts, range(1, minute + 1)
            )
            if current_bars is None or len(current_bars) < remaining:
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

            recent_bars = current_bars[minute - remaining :]
            if len(recent_bars) != remaining:
                exclusions[minute]["missing_bars"] += 1
                continue
            sigma_rec = mean_parkinson_variance(recent_bars).sqrt()
            sigma_all = mean_parkinson_variance(current_bars).sqrt()
            p_fast = terminal_win_probability(
                price=price,
                floor=floor,
                cap=cap,
                sigma=sigma_rec,
                remaining=remaining,
            )
            p_slow = terminal_win_probability(
                price=price,
                floor=floor,
                cap=cap,
                sigma=sigma_all,
                remaining=remaining,
            )
            if p_fast < p_star:
                abstentions[minute]["low_quality"] += 1
                continue
            if p_slow > q_star:
                abstentions[minute]["already_priced"] += 1
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
        p_star=p_star,
        q_star=q_star,
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
