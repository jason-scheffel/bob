# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import httpx

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXBTC"
PAGE_LIMIT = 1000
_ET = ZoneInfo("America/New_York")
_MONTH_ABBR = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)
_MAX_429_ATTEMPTS = 8
_INITIAL_429_DELAY_S = 0.5
_MAX_429_DELAY_S = 30.0


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


def kxbtc_event_ticker(close_ts: datetime) -> str:
    """KXBTC event ticker for an hourly close in America/New_York."""
    local = _as_utc(close_ts).astimezone(_ET)
    return (
        f"{SERIES_TICKER}-"
        f"{local.year % 100:02d}"
        f"{_MONTH_ABBR[local.month - 1]}"
        f"{local.day:02d}"
        f"{local.hour:02d}"
    )


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


def _unix(value: datetime) -> int:
    return int(_as_utc(value).timestamp())


def _hourly_closes(start: datetime, end: datetime) -> Iterator[datetime]:
    """Yield America/New_York hour boundaries with close_ts in ``[start, end)``."""
    start_utc = _as_utc(start)
    end_utc = _as_utc(end)
    local = start_utc.astimezone(_ET).replace(
        minute=0, second=0, microsecond=0
    )
    if local.astimezone(timezone.utc) < start_utc:
        local += timedelta(hours=1)
    while True:
        close = local.astimezone(timezone.utc)
        if close >= end_utc:
            return
        yield close
        local += timedelta(hours=1)


class KalshiClient:
    """Public Kalshi market client.

    Pass an ``httpx.Client`` with ``base_url=BASE_URL`` and an explicit
    ``timeout`` (auth headers can be added later on the same client).
    """

    def __init__(
        self,
        http: httpx.Client,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http
        self._sleep = sleep

    def fetch_settled_kxbtc(
        self,
        *,
        start: datetime,
        end: datetime,
    ) -> tuple[SettledEvent, ...]:
        start_utc = _as_utc(start)
        end_utc = _as_utc(end)
        if start_utc >= end_utc:
            raise ValueError("start must be earlier than end")
        markets = self._fetch_settled_markets(start_utc, end_utc)
        return parse_settled_kxbtc(markets, start=start_utc, end=end_utc)

    def _fetch_settled_markets(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        by_ticker: dict[str, dict[str, Any]] = {}
        # Live /markets: close_ts filters (not compatible with status=settled).
        # min is "after" so pass start-1 to keep closes at start; client
        # still enforces half-open [start, end).
        for market in self._iter_markets(
            "/markets",
            {
                "series_ticker": SERIES_TICKER,
                "min_close_ts": str(_unix(start) - 1),
                "max_close_ts": str(_unix(end)),
            },
        ):
            by_ticker[market["ticker"]] = market

        cutoff = self._market_settled_cutoff()
        if start < cutoff:
            hist_end = min(end, cutoff)
            for close in _hourly_closes(start, hist_end):
                for market in self._iter_markets(
                    "/historical/markets",
                    {"event_ticker": kxbtc_event_ticker(close)},
                ):
                    by_ticker.setdefault(market["ticker"], market)
        return list(by_ticker.values())

    def _market_settled_cutoff(self) -> datetime:
        payload = self._get_json("/historical/cutoff")
        return _parse_utc(payload["market_settled_ts"])

    def _iter_markets(
        self, path: str, params: Mapping[str, str]
    ) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            query = {**params, "limit": str(PAGE_LIMIT)}
            if cursor:
                query["cursor"] = cursor
            payload = self._get_json(path, params=query)
            for market in payload.get("markets", []):
                if isinstance(market, dict):
                    yield market
            cursor = payload.get("cursor") or None
            if not cursor:
                break

    def _get_json(
        self, path: str, *, params: Mapping[str, str] | None = None
    ) -> Any:
        delay = _INITIAL_429_DELAY_S
        for attempt in range(_MAX_429_ATTEMPTS):
            response = self._http.get(path, params=params)
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()
            if attempt == _MAX_429_ATTEMPTS - 1:
                response.raise_for_status()
            self._sleep(delay)
            delay = min(delay * 2, _MAX_429_DELAY_S)
        raise AssertionError("unreachable")
