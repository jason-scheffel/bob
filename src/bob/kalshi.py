# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXBTC"
PAGE_LIMIT = 1000


@dataclass(frozen=True, slots=True)
class Event:
    event_ticker: str
    close_ts: datetime
    expiration_value: Decimal


@dataclass(frozen=True, slots=True)
class Bracket:
    ticker: str
    event_ticker: str
    floor_strike: Decimal | None
    cap_strike: Decimal | None
    won: bool


@dataclass(frozen=True, slots=True)
class SettledEvent:
    event: Event
    brackets: tuple[Bracket, ...]


class KalshiParseError(ValueError):
    pass


def parse_settled_kxbtc(
    markets: Iterable[Mapping[str, Any]],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[SettledEvent, ...]:
    """Group finalized KXBTC markets into settled events.

    Date filter uses close_time in ``[start, end)`` when bounds are set.
    """
    by_event: dict[str, list[Mapping[str, Any]]] = {}
    for market in markets:
        event_ticker = market["event_ticker"]
        if not isinstance(event_ticker, str) or not event_ticker.startswith(
            f"{SERIES_TICKER}-"
        ):
            continue
        by_event.setdefault(event_ticker, []).append(market)

    start_utc = _as_utc(start) if start is not None else None
    end_utc = _as_utc(end) if end is not None else None

    settled: list[SettledEvent] = []
    for event_ticker, rows in by_event.items():
        close_ts = _event_close_ts(event_ticker, rows)
        if start_utc is not None and close_ts < start_utc:
            continue
        if end_utc is not None and close_ts >= end_utc:
            continue
        settled.append(_parse_event(event_ticker, rows, close_ts=close_ts))

    settled.sort(key=lambda item: (item.event.close_ts, item.event.event_ticker))
    return tuple(settled)


def _event_close_ts(
    event_ticker: str, rows: list[Mapping[str, Any]]
) -> datetime:
    if not rows:
        raise KalshiParseError(f"{event_ticker}: no markets")
    close_times = {_parse_utc(row["close_time"]) for row in rows}
    if len(close_times) != 1:
        raise KalshiParseError(f"{event_ticker}: inconsistent close_time")
    return close_times.pop()


def _parse_event(
    event_ticker: str,
    rows: list[Mapping[str, Any]],
    *,
    close_ts: datetime | None = None,
) -> SettledEvent:
    if close_ts is None:
        close_ts = _event_close_ts(event_ticker, rows)

    expiration_values = {str(row["expiration_value"]) for row in rows}
    if len(expiration_values) != 1:
        raise KalshiParseError(f"{event_ticker}: inconsistent expiration_value")
    expiration_raw = expiration_values.pop()
    if not expiration_raw:
        raise KalshiParseError(f"{event_ticker}: empty expiration_value")
    expiration_value = Decimal(expiration_raw)

    tickers: set[str] = set()
    brackets: list[Bracket] = []
    winners = 0
    for row in rows:
        if row.get("status") != "finalized":
            raise KalshiParseError(
                f"{event_ticker}: market {row.get('ticker')!r} not finalized"
            )
        ticker = row["ticker"]
        if not isinstance(ticker, str):
            raise KalshiParseError(f"{event_ticker}: missing ticker")
        if ticker in tickers:
            raise KalshiParseError(f"{event_ticker}: duplicate ticker {ticker}")
        tickers.add(ticker)

        result = row.get("result")
        if result not in ("yes", "no"):
            raise KalshiParseError(
                f"{event_ticker}: market {ticker} has invalid result {result!r}"
            )
        won = result == "yes"
        if won:
            winners += 1

        brackets.append(
            Bracket(
                ticker=ticker,
                event_ticker=event_ticker,
                floor_strike=_optional_decimal(row.get("floor_strike")),
                cap_strike=_optional_decimal(row.get("cap_strike")),
                won=won,
            )
        )

    if winners != 1:
        raise KalshiParseError(
            f"{event_ticker}: expected exactly one winner, got {winners}"
        )

    brackets.sort(
        key=lambda b: (
            b.floor_strike is None,
            b.floor_strike if b.floor_strike is not None else Decimal(0),
            b.cap_strike is None,
            b.cap_strike if b.cap_strike is not None else Decimal("Infinity"),
            b.ticker,
        )
    )
    return SettledEvent(
        event=Event(
            event_ticker=event_ticker,
            close_ts=close_ts,
            expiration_value=expiration_value,
        ),
        brackets=tuple(brackets),
    )


def _optional_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise KalshiParseError(f"invalid timestamp: {value!r}")
    # fromisoformat handles trailing Z in 3.11+
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"datetime must be timezone-aware: {value!r}")
    return value.astimezone(timezone.utc)


class KalshiClient:
    """Public Kalshi market client.

    Pass an ``httpx.Client`` with ``base_url=BASE_URL`` and an explicit
    ``timeout`` (auth headers can be added later on the same client).
    """

    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def fetch_settled_kxbtc(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[SettledEvent, ...]:
        markets = self._fetch_all_settled_markets()
        return parse_settled_kxbtc(markets, start=start, end=end)

    def _fetch_all_settled_markets(self) -> list[dict[str, Any]]:
        by_ticker: dict[str, dict[str, Any]] = {}
        for market in self._iter_markets(
            "/markets",
            {"series_ticker": SERIES_TICKER, "status": "settled"},
        ):
            by_ticker[market["ticker"]] = market
        for market in self._iter_markets(
            "/historical/markets",
            {"series_ticker": SERIES_TICKER},
        ):
            by_ticker.setdefault(market["ticker"], market)
        return list(by_ticker.values())

    def _iter_markets(
        self, path: str, params: Mapping[str, str]
    ) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            query = {**params, "limit": str(PAGE_LIMIT)}
            if cursor:
                query["cursor"] = cursor
            response = self._http.get(path, params=query)
            response.raise_for_status()
            payload = response.json()
            for market in payload.get("markets", []):
                if isinstance(market, dict):
                    yield market
            cursor = payload.get("cursor") or None
            if not cursor:
                break
