# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s15: impulse–flag arrival current-bracket hold.

Buy YES/NO on the live closed bracket after a material net move from the
hour open that has finished dwelling (last K closes stay in-bracket).

Treat this module as frozen. Prefer a new strategy module over editing s15.
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
from bob.research.trades import TradeObservation

STRATEGY = "s15"
STRATEGY_SUMMARY = "impulse-flag arrival current bracket"

DEFAULT_CHECKPOINT_MINUTES = (40, 45, 50, 55)
MIN_CHECKPOINT = 35
DEFAULT_MOVE = Decimal("250")
DEFAULT_DWELL = 4

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
    "small_move",
    "in_transit",
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
    move: Decimal
    dwell: int
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


def _close_bracket(connection, close_ts, brackets, minute):
    bar = load_candle_ohlc(connection, checkpoint_end_ts(close_ts, minute))
    if bar is None:
        return None, "missing_bars"
    price = finite_decimal(bar.close)
    if price is None:
        return None, "bad_price"
    matches = brackets_containing(price, brackets)
    reason = _classify(len(matches))
    if reason is not None:
        return None, reason
    return (matches[0], price), None


def evaluate(
    connection: sqlite3.Connection,
    *,
    minutes: Sequence[int] = DEFAULT_CHECKPOINT_MINUTES,
    side: Side = "yes",
    move: Decimal = DEFAULT_MOVE,
    dwell: int = DEFAULT_DWELL,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    """Score impulse–flag arrival hold accuracy."""
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    if not isinstance(move, Decimal):
        move = Decimal(str(move))
    if not move.is_finite() or move <= 0:
        raise ValueError(f"move must be a positive decimal, got {move}")
    if not isinstance(dwell, int) or dwell < 2:
        raise ValueError(f"dwell must be an int >= 2, got {dwell!r}")
    minute_list = tuple(minutes)
    if not minute_list:
        raise ValueError("minutes must be non-empty")
    if len(set(minute_list)) != len(minute_list):
        raise ValueError("minutes must be unique")
    for minute in minute_list:
        if not MIN_CHECKPOINT <= minute <= 59:
            raise ValueError(f"minute must be in {MIN_CHECKPOINT}..59, got {minute}")
        if minute - dwell + 1 < 1:
            raise ValueError(
                f"minute {minute} needs dwell={dwell} closes starting at minute ≥ 1"
            )
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

            hit, fail = _close_bracket(
                connection, event.close_ts, brackets, minute
            )
            if hit is None:
                assert fail is not None
                if fail in ("missing_bars", "bad_price"):
                    exclusions[minute][fail] += 1
                else:
                    exclusions[minute][fail] += 1
                continue
            selected, price = hit
            if selected.floor_strike is None or selected.cap_strike is None:
                abstentions[minute]["open_ended_bracket"] += 1
                continue

            if abs(price - open_px) < move:
                abstentions[minute]["small_move"] += 1
                continue

            dwell_ok = True
            for dwell_minute in range(minute - dwell + 1, minute + 1):
                dwell_hit, dwell_fail = _close_bracket(
                    connection, event.close_ts, brackets, dwell_minute
                )
                if dwell_hit is None:
                    assert dwell_fail is not None
                    exclusions[minute][dwell_fail] += 1
                    dwell_ok = False
                    break
                dwell_bracket, _ = dwell_hit
                if dwell_bracket.ticker != selected.ticker:
                    abstentions[minute]["in_transit"] += 1
                    dwell_ok = False
                    break
            if not dwell_ok:
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
        move=move,
        dwell=dwell,
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
