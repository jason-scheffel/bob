# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s17: sticky-current modal occupancy hold.

Buy YES/NO on the live closed bracket when it is the unique modal close
bracket over minutes 1..M with occupancy ≥ ``min_occupancy``, and the
hour has not made a large open-to-close move. Unlike s7 (which can buy a
modal you are not sitting in), this requires mode == current.

Treat this module as frozen. Prefer a new strategy module over editing s17.
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
    load_minute_closes,
)
from bob.research.trades import TradeObservation

STRATEGY = "s17"
STRATEGY_SUMMARY = "sticky-current modal occupancy"

DEFAULT_CHECKPOINT_MINUTES = (40, 45, 50, 55)
MIN_CHECKPOINT = 20
DEFAULT_MIN_OCCUPANCY = Decimal("0.60")
DEFAULT_MAX_MOVE = Decimal("250")

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
    "no_unique_mode",
    "mode_not_current",
    "low_occupancy",
    "large_move",
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
    min_occupancy: Decimal
    max_move: Decimal
    minutes: tuple[MinuteStats, ...]
    trades: tuple[TradeObservation, ...]


def _outcome_win(*, selected_won: bool, side: Side) -> bool:
    return selected_won if side == "yes" else not selected_won


def evaluate(
    connection: sqlite3.Connection,
    *,
    minutes: Sequence[int] = DEFAULT_CHECKPOINT_MINUTES,
    side: Side = "yes",
    min_occupancy: Decimal = DEFAULT_MIN_OCCUPANCY,
    max_move: Decimal = DEFAULT_MAX_MOVE,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    """Score sticky-current modal occupancy hold accuracy."""
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    if not isinstance(min_occupancy, Decimal):
        min_occupancy = Decimal(str(min_occupancy))
    if not isinstance(max_move, Decimal):
        max_move = Decimal(str(max_move))
    if not min_occupancy.is_finite() or not (
        Decimal("0") < min_occupancy <= Decimal("1")
    ):
        raise ValueError(f"min_occupancy must be in (0, 1], got {min_occupancy}")
    if not max_move.is_finite() or max_move <= 0:
        raise ValueError(f"max_move must be a positive decimal, got {max_move}")
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

            open_bar = load_candle_ohlc(
                connection, checkpoint_end_ts(event.close_ts, 1)
            )
            if open_bar is None:
                exclusions[minute]["missing_bars"] += 1
                continue
            open_px = finite_decimal(open_bar.open)
            if open_px is None:
                exclusions[minute]["bad_price"] += 1
                continue

            closes = load_minute_closes(
                connection, event.close_ts, range(1, minute + 1)
            )
            if closes is None:
                exclusions[minute]["missing_bars"] += 1
                continue

            tickers: list[str] = []
            mapping_ok = True
            for close in closes:
                matches = brackets_containing(close, brackets)
                if len(matches) != 1:
                    exclusions[minute][
                        "no_bracket_match"
                        if len(matches) == 0
                        else "ambiguous_bracket"
                    ] += 1
                    mapping_ok = False
                    break
                tickers.append(matches[0].ticker)
            if not mapping_ok:
                continue

            current = brackets_containing(closes[-1], brackets)[0]
            if current.floor_strike is None or current.cap_strike is None:
                abstentions[minute]["open_ended_bracket"] += 1
                continue

            counts = Counter(tickers)
            top_count = counts.most_common(1)[0][1]
            leaders = [ticker for ticker, count in counts.items() if count == top_count]
            if len(leaders) != 1:
                abstentions[minute]["no_unique_mode"] += 1
                continue
            if leaders[0] != current.ticker:
                abstentions[minute]["mode_not_current"] += 1
                continue
            occupancy = Decimal(top_count) / Decimal(minute)
            if occupancy < min_occupancy:
                abstentions[minute]["low_occupancy"] += 1
                continue
            if abs(closes[-1] - open_px) >= max_move:
                abstentions[minute]["large_move"] += 1
                continue

            end_ts = checkpoint_end_ts(event.close_ts, minute)
            won = _outcome_win(selected_won=current.won, side=side)
            if won:
                wins[minute] += 1
            else:
                losses[minute] += 1
            trades.append(
                TradeObservation(
                    event_ticker=event.event_ticker,
                    market_ticker=current.ticker,
                    minute=minute,
                    end_ts=end_ts,
                    side=side,
                    won=won,
                )
            )

    return Report(
        strategy=STRATEGY,
        side=side,
        min_occupancy=min_occupancy,
        max_move=max_move,
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
