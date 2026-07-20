# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Quote-sim gross P&L scoring from trade observations + market_candles.

Strategies never read quotes; this layer prices already-selected trades.
Labels are quote-sim: minute close ≠ proven executable depth at the boundary.
Gross only (settlement − premium); fees are out of scope.

Optional early-exit overlays (first touch wins; same bar prefers stop):
- stop-bid: exit when side mark ≤ stop_bid (YES@bid / NO@(1−ask))
- take-pct: exit when side mark ≥ premium × (1 + take_pct)
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from bob.db import load_market_quote
from bob.research.common import finite_decimal
from bob.research.trades import Side, TradeObservation

ONE = Decimal("1")
ZERO = Decimal("0")
DEFAULT_STOP_FROM = 55

ExitKind = Literal["stop", "take"]


@dataclass(frozen=True, slots=True)
class QuoteSimTrade:
    observation: TradeObservation
    premium: Decimal
    settlement: Decimal
    gross: Decimal
    stopped: bool = False
    taken: bool = False
    exit_minute: int | None = None


@dataclass(frozen=True, slots=True)
class QuoteSimReport:
    """Gross quote-sim P&L aggregates for one strategy evaluation."""

    strategy_eligible: int
    quote_excluded: int
    quote_eligible: int
    premium: Decimal
    payout: Decimal
    gross: Decimal
    wins: int
    losses: int
    trades: tuple[QuoteSimTrade, ...]
    stopped: int = 0
    taken: int = 0

    @property
    def win_rate(self) -> float | None:
        if self.quote_eligible == 0:
            return None
        return self.wins / self.quote_eligible

    @property
    def return_on_premium(self) -> Decimal | None:
        if self.premium == 0:
            return None
        return self.gross / self.premium


def validate_quote(
    yes_bid_close: str | None,
    yes_ask_close: str | None,
) -> tuple[Decimal, Decimal] | None:
    """Return ``(bid, ask)`` when ``0 ≤ bid ≤ ask ≤ 1``; else None."""
    if yes_bid_close is None or yes_ask_close is None:
        return None
    bid = finite_decimal(yes_bid_close)
    ask = finite_decimal(yes_ask_close)
    if bid is None or ask is None:
        return None
    if bid < ZERO or ask > ONE or bid > ask:
        return None
    return bid, ask


def premium_for_side(side: Side, bid: Decimal, ask: Decimal) -> Decimal:
    if side == "yes":
        return ask
    return ONE - bid


def exit_mark_for_side(side: Side, bid: Decimal, ask: Decimal) -> Decimal:
    """Mark used to sell the purchased side (YES@bid / NO@(1−ask))."""
    if side == "yes":
        return bid
    return ONE - ask


def settlement_payout(*, won: bool) -> Decimal:
    """``won`` already means the purchased side paid out."""
    return ONE if won else ZERO


def _as_decimal(value: Decimal | str | int | float) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _validate_exit_from(exit_from: int) -> None:
    if not isinstance(exit_from, int) or not 1 <= exit_from <= 59:
        raise ValueError(f"stop_from must be an int in 1..59, got {exit_from!r}")


def _validate_stop_bid(stop_bid: Decimal | None) -> Decimal | None:
    if stop_bid is None:
        return None
    stop_bid = _as_decimal(stop_bid)
    if not stop_bid.is_finite() or stop_bid < ZERO or stop_bid > ONE:
        raise ValueError(f"stop_bid must be a Decimal in [0, 1], got {stop_bid}")
    return stop_bid


def _validate_take_pct(take_pct: Decimal | None) -> Decimal | None:
    if take_pct is None:
        return None
    take_pct = _as_decimal(take_pct)
    if not take_pct.is_finite() or take_pct <= ZERO:
        raise ValueError(f"take_pct must be a positive Decimal, got {take_pct}")
    return take_pct


def find_early_exit(
    connection: sqlite3.Connection,
    obs: TradeObservation,
    *,
    premium: Decimal,
    stop_bid: Decimal | None,
    take_pct: Decimal | None,
    exit_from: int,
) -> tuple[Decimal, int, ExitKind] | None:
    """First post-entry stop or take from ``exit_from``..59, if any."""
    if stop_bid is None and take_pct is None:
        return None
    take_level = None if take_pct is None else premium * (ONE + take_pct)
    start = max(obs.minute + 1, exit_from)
    for minute in range(start, 60):
        end_ts = obs.end_ts + (minute - obs.minute) * 60
        quote = load_market_quote(connection, obs.market_ticker, end_ts)
        if quote is None:
            continue
        validated = validate_quote(quote.yes_bid_close, quote.yes_ask_close)
        if validated is None:
            continue
        bid, ask = validated
        mark = exit_mark_for_side(obs.side, bid, ask)
        hit_stop = stop_bid is not None and mark <= stop_bid
        hit_take = take_level is not None and mark >= take_level
        if hit_stop:
            return mark, minute, "stop"
        if hit_take:
            return mark, minute, "take"
    return None


def find_stop_exit(
    connection: sqlite3.Connection,
    obs: TradeObservation,
    *,
    stop_bid: Decimal,
    stop_from: int,
) -> tuple[Decimal, int] | None:
    """Compatibility wrapper: stop-only early exit."""
    # Premium unused for stop-only; pass ZERO.
    hit = find_early_exit(
        connection,
        obs,
        premium=ZERO,
        stop_bid=stop_bid,
        take_pct=None,
        exit_from=stop_from,
    )
    if hit is None:
        return None
    mark, minute, _kind = hit
    return mark, minute


def score_trades(
    connection: sqlite3.Connection,
    observations: Sequence[TradeObservation],
    *,
    stop_bid: Decimal | None = None,
    take_pct: Decimal | None = None,
    stop_from: int = DEFAULT_STOP_FROM,
) -> QuoteSimReport:
    """Price observations at their checkpoint quote closes (1 contract)."""
    _validate_exit_from(stop_from)
    stop_bid = _validate_stop_bid(stop_bid)
    take_pct = _validate_take_pct(take_pct)
    priced: list[QuoteSimTrade] = []
    quote_excluded = 0
    for obs in observations:
        quote = load_market_quote(connection, obs.market_ticker, obs.end_ts)
        if quote is None:
            quote_excluded += 1
            continue
        validated = validate_quote(quote.yes_bid_close, quote.yes_ask_close)
        if validated is None:
            quote_excluded += 1
            continue
        bid, ask = validated
        premium = premium_for_side(obs.side, bid, ask)
        stopped = False
        taken = False
        exit_minute: int | None = None
        if stop_bid is not None or take_pct is not None:
            hit = find_early_exit(
                connection,
                obs,
                premium=premium,
                stop_bid=stop_bid,
                take_pct=take_pct,
                exit_from=stop_from,
            )
            if hit is not None:
                settlement, exit_minute, kind = hit
                stopped = kind == "stop"
                taken = kind == "take"
            else:
                settlement = settlement_payout(won=obs.won)
        else:
            settlement = settlement_payout(won=obs.won)
        priced.append(
            QuoteSimTrade(
                observation=obs,
                premium=premium,
                settlement=settlement,
                gross=settlement - premium,
                stopped=stopped,
                taken=taken,
                exit_minute=exit_minute,
            )
        )

    premium_sum = sum((trade.premium for trade in priced), ZERO)
    payout_sum = sum((trade.settlement for trade in priced), ZERO)
    gross_sum = sum((trade.gross for trade in priced), ZERO)
    wins = sum(
        1
        for trade in priced
        if trade.taken or (not trade.stopped and trade.observation.won)
    )
    losses = len(priced) - wins
    return QuoteSimReport(
        strategy_eligible=len(observations),
        quote_excluded=quote_excluded,
        quote_eligible=len(priced),
        premium=premium_sum,
        payout=payout_sum,
        gross=gross_sum,
        wins=wins,
        losses=losses,
        trades=tuple(priced),
        stopped=sum(1 for trade in priced if trade.stopped),
        taken=sum(1 for trade in priced if trade.taken),
    )


def score_trades_by_minute(
    connection: sqlite3.Connection,
    observations: Sequence[TradeObservation],
    minutes: Sequence[int],
    *,
    stop_bid: Decimal | None = None,
    take_pct: Decimal | None = None,
    stop_from: int = DEFAULT_STOP_FROM,
) -> tuple[tuple[int, QuoteSimReport], ...]:
    """Score separately for each checkpoint minute (order follows ``minutes``)."""
    by_minute = {
        minute: [obs for obs in observations if obs.minute == minute]
        for minute in minutes
    }
    return tuple(
        (
            minute,
            score_trades(
                connection,
                by_minute.get(minute, ()),
                stop_bid=stop_bid,
                take_pct=take_pct,
                stop_from=stop_from,
            ),
        )
        for minute in minutes
    )
