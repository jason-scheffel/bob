# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Quote-sim gross P&L scoring from trade observations + market_candles.

Strategies never read quotes; this layer prices already-selected trades.
Labels are quote-sim: minute close ≠ proven executable depth at the boundary.
Gross only (settlement − premium); fees are out of scope.

Optional stop-bid overlay: after entry, from ``stop_from`` onward, exit at
the first minute whose side mark ≤ ``stop_bid`` (YES@bid / NO@(1−ask)).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from bob.db import load_market_quote
from bob.research.common import finite_decimal
from bob.research.trades import Side, TradeObservation

ONE = Decimal("1")
ZERO = Decimal("0")
DEFAULT_STOP_FROM = 55


@dataclass(frozen=True, slots=True)
class QuoteSimTrade:
    observation: TradeObservation
    premium: Decimal
    settlement: Decimal
    gross: Decimal
    stopped: bool = False
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


def _validate_stop_params(
    stop_bid: Decimal | None,
    stop_from: int,
) -> Decimal | None:
    if stop_bid is None:
        return None
    if not isinstance(stop_bid, Decimal):
        stop_bid = Decimal(str(stop_bid))
    if not stop_bid.is_finite() or stop_bid < ZERO or stop_bid > ONE:
        raise ValueError(f"stop_bid must be a Decimal in [0, 1], got {stop_bid}")
    if not isinstance(stop_from, int) or not 1 <= stop_from <= 59:
        raise ValueError(f"stop_from must be an int in 1..59, got {stop_from!r}")
    return stop_bid


def find_stop_exit(
    connection: sqlite3.Connection,
    obs: TradeObservation,
    *,
    stop_bid: Decimal,
    stop_from: int,
) -> tuple[Decimal, int] | None:
    """First post-entry mark ≤ stop_bid from ``stop_from``..59, if any."""
    start = max(obs.minute + 1, stop_from)
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
        if mark <= stop_bid:
            return mark, minute
    return None


def score_trades(
    connection: sqlite3.Connection,
    observations: Sequence[TradeObservation],
    *,
    stop_bid: Decimal | None = None,
    stop_from: int = DEFAULT_STOP_FROM,
) -> QuoteSimReport:
    """Price observations at their checkpoint quote closes (1 contract)."""
    stop_bid = _validate_stop_params(stop_bid, stop_from)
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
        exit_minute: int | None = None
        if stop_bid is not None:
            hit = find_stop_exit(
                connection,
                obs,
                stop_bid=stop_bid,
                stop_from=stop_from,
            )
            if hit is not None:
                settlement, exit_minute = hit
                stopped = True
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
                exit_minute=exit_minute,
            )
        )

    premium_sum = sum((trade.premium for trade in priced), ZERO)
    payout_sum = sum((trade.settlement for trade in priced), ZERO)
    gross_sum = sum((trade.gross for trade in priced), ZERO)
    wins = sum(
        1 for trade in priced if not trade.stopped and trade.observation.won
    )
    losses = len(priced) - wins
    stopped_count = sum(1 for trade in priced if trade.stopped)
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
        stopped=stopped_count,
    )


def score_trades_by_minute(
    connection: sqlite3.Connection,
    observations: Sequence[TradeObservation],
    minutes: Sequence[int],
    *,
    stop_bid: Decimal | None = None,
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
                stop_from=stop_from,
            ),
        )
        for minute in minutes
    )
