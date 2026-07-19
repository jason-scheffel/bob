# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Trade observations recorded by research strategies for quote-sim P&L."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["yes", "no"]


@dataclass(frozen=True, slots=True)
class TradeObservation:
    """One strategy-eligible hold-to-settlement trade at a checkpoint."""

    event_ticker: str
    market_ticker: str
    minute: int
    end_ts: int
    side: Side
    won: bool
