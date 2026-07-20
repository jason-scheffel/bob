# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""s20: walk-forward path-analog continuation vote.

At each checkpoint, match prior hours on normalized bracket position,
trailing range dwell, and normalized excursion; apply their realized
continuations; buy YES/NO on the unique plurality destination bracket.

Treat this module as frozen. Prefer a new strategy module over editing s20.
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

from bob.browse import BracketRow, load_brackets, winning_bracket
from bob.research.common import (
    brackets_containing,
    checkpoint_end_ts,
    closed_bracket_width,
    finite_decimal,
    load_all_complete_events,
    load_candle_ohlc,
)
from bob.research.trades import TradeObservation

STRATEGY = "s20"
STRATEGY_SUMMARY = "walk-forward path-analog continuation vote"

DEFAULT_CHECKPOINT_MINUTES = (40, 45, 50, 55)
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
    "insufficient_analogs",
    "tied_plurality",
    "no_valid_votes",
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


@dataclass(frozen=True, slots=True)
class _Analog:
    features: tuple[float, float, float]
    continuation: float
    close_ts: datetime
    event_ticker: str


@dataclass(frozen=True, slots=True)
class _PathState:
    features: tuple[float, float, float]
    continuation: float
    checkpoint_close: Decimal
    current: BracketRow
    width: Decimal


def trailing_minutes(checkpoint: int) -> range:
    """Trailing bars of length ``60 - checkpoint``, ending at ``checkpoint``."""
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


def _manhattan(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return abs(left[0] - right[0]) + abs(left[1] - right[1]) + abs(left[2] - right[2])


def _unique_plurality(votes: Counter[str]) -> str | None:
    if not votes:
        return None
    best = max(votes.values())
    winners = [ticker for ticker, count in votes.items() if count == best]
    if len(winners) != 1:
        return None
    return winners[0]


def _range_in_bracket(
    high: Decimal,
    low: Decimal,
    floor: Decimal,
    cap: Decimal,
) -> bool:
    """Design dwell: low ≥ floor and high < cap."""
    return low >= floor and high < cap


def _load_path_state(
    connection: sqlite3.Connection,
    *,
    close_ts: datetime,
    minute: int,
    brackets: Sequence[BracketRow],
) -> _PathState | ExclusionReason | AbstentionReason:
    bars: list[tuple[Decimal, Decimal, Decimal]] = []
    for window_minute in trailing_minutes(minute):
        bar = load_candle_ohlc(
            connection, checkpoint_end_ts(close_ts, window_minute)
        )
        if bar is None:
            return "missing_bars"
        high = finite_decimal(bar.high)
        low = finite_decimal(bar.low)
        close = finite_decimal(bar.close)
        if high is None or low is None or close is None:
            return "bad_price"
        bars.append((high, low, close))

    checkpoint_close = bars[-1][2]
    matches = brackets_containing(checkpoint_close, brackets)
    reason = _classify(len(matches))
    if reason is not None:
        return reason
    current = matches[0]
    if current.floor_strike is None or current.cap_strike is None:
        return "open_ended_bracket"
    width = closed_bracket_width(current.floor_strike, current.cap_strike)
    if width is None:
        return "bad_price"
    floor = finite_decimal(current.floor_strike)
    cap = finite_decimal(current.cap_strike)
    if floor is None or cap is None:
        return "bad_price"

    terminal_bar = load_candle_ohlc(connection, int(close_ts.timestamp()))
    if terminal_bar is None:
        return "missing_bars"
    terminal_close = finite_decimal(terminal_bar.close)
    if terminal_close is None:
        return "bad_price"

    contained = sum(
        1 for high, low, _ in bars if _range_in_bracket(high, low, floor, cap)
    )
    max_high = max(high for high, _, _ in bars)
    min_low = min(low for _, low, _ in bars)
    features = (
        float((checkpoint_close - floor) / width),
        contained / len(bars),
        float((max_high - min_low) / width),
    )
    continuation = float((terminal_close - checkpoint_close) / width)
    return _PathState(
        features=features,
        continuation=continuation,
        checkpoint_close=checkpoint_close,
        current=current,
        width=width,
    )


def _vote_destination(
    *,
    state: _PathState,
    prior: Sequence[_Analog],
    brackets: Sequence[BracketRow],
) -> BracketRow | AbstentionReason:
    k = math.isqrt(len(prior))
    if k == 0:
        return "insufficient_analogs"
    nearest = sorted(
        prior,
        key=lambda analog: (
            _manhattan(state.features, analog.features),
            analog.close_ts,
            analog.event_ticker,
        ),
    )[:k]
    votes: Counter[str] = Counter()
    by_ticker = {bracket.ticker: bracket for bracket in brackets}
    for analog in nearest:
        target = state.checkpoint_close + Decimal(str(analog.continuation)) * state.width
        destinations = brackets_containing(target, brackets)
        if len(destinations) != 1:
            continue
        destination = destinations[0]
        if destination.floor_strike is None or destination.cap_strike is None:
            continue
        votes[destination.ticker] += 1
    if not votes:
        return "no_valid_votes"
    selected_ticker = _unique_plurality(votes)
    if selected_ticker is None:
        return "tied_plurality"
    return by_ticker[selected_ticker]


def evaluate(
    connection: sqlite3.Connection,
    *,
    minutes: Sequence[int] = DEFAULT_CHECKPOINT_MINUTES,
    side: Side = "yes",
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    """Score walk-forward path-analog continuation accuracy."""
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

    events = load_all_complete_events(connection)
    wins = Counter({minute: 0 for minute in minute_list})
    losses = Counter({minute: 0 for minute in minute_list})
    exclusions: dict[int, Counter[str]] = {minute: Counter() for minute in minute_list}
    abstentions: dict[int, Counter[str]] = {minute: Counter() for minute in minute_list}
    trades: list[TradeObservation] = []
    analogs: dict[int, list[_Analog]] = {minute: [] for minute in minute_list}

    for event in events:
        in_window = start is None or (start <= event.close_ts < end)
        brackets = load_brackets(connection, event.event_ticker)
        winner = winning_bracket(brackets)
        expiration = finite_decimal(event.expiration_value)

        for minute in minute_list:
            path = _load_path_state(
                connection,
                close_ts=event.close_ts,
                minute=minute,
                brackets=brackets,
            )

            if in_window:
                if winner is None:
                    exclusions[minute]["no_winner"] += 1
                elif expiration is None:
                    exclusions[minute]["bad_expiration"] += 1
                else:
                    settlement = brackets_containing(expiration, brackets)
                    if len(settlement) != 1 or settlement[0] != winner:
                        exclusions[minute]["bad_winner_invariant"] += 1
                    elif isinstance(path, str):
                        if path in (
                            "open_ended_bracket",
                            "insufficient_analogs",
                            "tied_plurality",
                            "no_valid_votes",
                        ):
                            abstentions[minute][path] += 1
                        else:
                            exclusions[minute][path] += 1
                    else:
                        decision = _vote_destination(
                            state=path,
                            prior=analogs[minute],
                            brackets=brackets,
                        )
                        if isinstance(decision, str):
                            abstentions[minute][decision] += 1
                        else:
                            end_ts = checkpoint_end_ts(event.close_ts, minute)
                            won = _outcome_win(selected_won=decision.won, side=side)
                            if won:
                                wins[minute] += 1
                            else:
                                losses[minute] += 1
                            trades.append(
                                TradeObservation(
                                    event_ticker=event.event_ticker,
                                    market_ticker=decision.ticker,
                                    minute=minute,
                                    end_ts=end_ts,
                                    side=side,
                                    won=won,
                                )
                            )

            if isinstance(path, _PathState):
                analogs[minute].append(
                    _Analog(
                        features=path.features,
                        continuation=path.continuation,
                        close_ts=event.close_ts,
                        event_ticker=event.event_ticker,
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
