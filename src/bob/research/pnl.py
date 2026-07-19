# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Quote-sim gross P&L scoring from trade observations + market_candles.

Strategies never read quotes; this layer prices already-selected trades.
Labels are quote-sim: minute close ≠ proven executable depth at the boundary.
Gross only (settlement − premium); fees are out of scope.
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


@dataclass(frozen=True, slots=True)
class QuoteSimTrade:
    observation: TradeObservation
    premium: Decimal
    settlement: Decimal
    gross: Decimal


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


def settlement_payout(*, won: bool) -> Decimal:
    """``won`` already means the purchased side paid out."""
    return ONE if won else ZERO


def score_trades(
    connection: sqlite3.Connection,
    observations: Sequence[TradeObservation],
) -> QuoteSimReport:
    """Price observations at their checkpoint quote closes (1 contract)."""
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
        settlement = settlement_payout(won=obs.won)
        priced.append(
            QuoteSimTrade(
                observation=obs,
                premium=premium,
                settlement=settlement,
                gross=settlement - premium,
            )
        )

    premium_sum = sum((trade.premium for trade in priced), ZERO)
    payout_sum = sum((trade.settlement for trade in priced), ZERO)
    gross_sum = sum((trade.gross for trade in priced), ZERO)
    wins = sum(1 for trade in priced if trade.observation.won)
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
    )


def score_trades_by_minute(
    connection: sqlite3.Connection,
    observations: Sequence[TradeObservation],
    minutes: Sequence[int],
) -> tuple[tuple[int, QuoteSimReport], ...]:
    """Score separately for each checkpoint minute (order follows ``minutes``)."""
    by_minute = {
        minute: [obs for obs in observations if obs.minute == minute]
        for minute in minutes
    }
    return tuple(
        (minute, score_trades(connection, by_minute.get(minute, ())))
        for minute in minutes
    )
