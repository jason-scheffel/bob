# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import base64
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXBTC"
BRTI_INDEX_ID = "BRTI"
CF_HISTORY_VALUES_PATH = "/cfbenchmarks/history/values"
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
# CF passthrough costs 50 read tokens/req (~4 rps on Basic). Combined backfill default.
DEFAULT_MAX_RPS = 3.0
# Market candlesticks are ordinary public GETs (not CF). Separate budget.
DEFAULT_MARKET_CANDLE_RPS = 20.0

STATUS_COMPLETE = "complete"
STATUS_MISSING_EXPIRATION = "missing_expiration_value"
STATUS_NO_MARKETS = "no_markets"


@dataclass(frozen=True, slots=True)
class Event:
    event_ticker: str
    close_ts: datetime
    status: str
    expiration_value: Decimal | None


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


class KalshiAuthError(RuntimeError):
    pass


class KalshiAPIError(RuntimeError):
    pass


class KalshiCredentialsError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class KalshiCredentials:
    api_key_id: str
    private_key: RSAPrivateKey
    base_url: str


@dataclass(frozen=True, slots=True)
class FetchUpdate:
    """Progress snapshot while fetching settled markets."""

    phase: str
    detail: str
    requests: int = 0
    markets: int = 0
    retries_429: int = 0
    completed: int | None = None
    total: int | None = None


class RateLimiter:
    """Space out requests to stay under a sustained requests-per-second budget.

    ``max_rps <= 0`` disables limiting. On 429, the effective rate is halved
    (floor ``min(1/s, max_rps)``) until successes climb it back toward
    ``max_rps``.
    """

    def __init__(
        self,
        max_rps: float,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_rps = max_rps
        self._rps = max_rps
        self._sleep = sleep
        self._monotonic = monotonic
        self._next_at = 0.0

    @property
    def rps(self) -> float:
        return self._rps

    def wait(self) -> None:
        if self._rps <= 0:
            return
        now = self._monotonic()
        delay = self._next_at - now
        if delay > 0:
            self._sleep(delay)
            now = self._monotonic()
        self._next_at = now + (1.0 / self._rps)

    def note_429(self) -> None:
        if self._max_rps <= 0:
            return
        floor = min(1.0, self._max_rps)
        self._rps = max(floor, self._rps / 2.0)

    def note_success(self) -> None:
        if self._max_rps <= 0 or self._rps >= self._max_rps:
            return
        self._rps = min(self._max_rps, self._rps * 1.25)


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


def expected_kxbtc_event_tickers(
    start: datetime, end: datetime
) -> frozenset[str]:
    """Event tickers for hourly KXBTC closes in ``[start, end)``."""
    return frozenset(
        kxbtc_event_ticker(close) for close in _hourly_closes(start, end)
    )


def parse_settled_kxbtc(
    markets: Iterable[Mapping[str, Any]],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    on_skip: Callable[[str, str], None] | None = None,
) -> tuple[SettledEvent, ...]:
    """Group finalized KXBTC markets into settled events.

    Date filter uses close_time in ``[start, end)`` when bounds are set.
    Parse failures call ``on_skip(event_ticker, reason)`` when provided;
    otherwise they raise.
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
        try:
            close_ts = _event_close_ts(event_ticker, rows)
            if start_utc is not None and close_ts < start_utc:
                continue
            if end_utc is not None and close_ts >= end_utc:
                continue
            settled.append(_parse_event(event_ticker, rows, close_ts=close_ts))
        except KalshiParseError as exc:
            if on_skip is None:
                raise
            on_skip(event_ticker, str(exc))

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


def _event_expiration_value(
    event_ticker: str, rows: list[Mapping[str, Any]]
) -> Decimal | None:
    # Some brackets arrive with a blank expiration_value; ignore blanks.
    values = {
        str(row["expiration_value"]).strip()
        for row in rows
        if str(row.get("expiration_value") or "").strip()
    }
    if len(values) == 1:
        return Decimal(values.pop())
    if not values:
        return None
    raise KalshiParseError(f"{event_ticker}: inconsistent expiration_value")


def _parse_brackets(
    event_ticker: str, rows: list[Mapping[str, Any]]
) -> tuple[Bracket, ...]:
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
    return tuple(brackets)


def _parse_event(
    event_ticker: str,
    rows: list[Mapping[str, Any]],
    *,
    close_ts: datetime | None = None,
) -> SettledEvent:
    if close_ts is None:
        close_ts = _event_close_ts(event_ticker, rows)

    brackets = _parse_brackets(event_ticker, rows)
    expiration_value = _event_expiration_value(event_ticker, rows)
    status = (
        STATUS_COMPLETE
        if expiration_value is not None
        else STATUS_MISSING_EXPIRATION
    )
    return SettledEvent(
        event=Event(
            event_ticker=event_ticker,
            close_ts=close_ts,
            status=status,
            expiration_value=expiration_value,
        ),
        brackets=brackets,
    )


def no_markets_event(event_ticker: str, close_ts: datetime) -> SettledEvent:
    """Observation that Kalshi returned no markets for an expected hour."""
    return SettledEvent(
        event=Event(
            event_ticker=event_ticker,
            close_ts=_as_utc(close_ts),
            status=STATUS_NO_MARKETS,
            expiration_value=None,
        ),
        brackets=(),
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
    """Kalshi market client (public) plus optional CF Benchmarks auth.

    Pass an ``httpx.Client`` with ``base_url`` matching credentials when CF
    routes are used. Settled-market GETs stay unsigned; CF history is signed.
    """

    def __init__(
        self,
        http: httpx.Client,
        *,
        credentials: KalshiCredentials | None = None,
        max_rps: float = DEFAULT_MAX_RPS,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        on_update: Callable[[FetchUpdate], None] | None = None,
    ) -> None:
        self._http = http
        self._credentials = credentials
        self._sleep = sleep
        self._on_update = on_update
        self._limiter = RateLimiter(max_rps, sleep=sleep, monotonic=monotonic)
        self._requests = 0
        self._retries_429 = 0
        self._markets = 0

    @property
    def requests(self) -> int:
        return self._requests

    @property
    def retries_429(self) -> int:
        return self._retries_429

    def fetch_settled_kxbtc(
        self,
        *,
        start: datetime,
        end: datetime,
        on_skip: Callable[[str, str], None] | None = None,
        only_event_tickers: frozenset[str] | None = None,
    ) -> tuple[SettledEvent, ...]:
        start_utc = _as_utc(start)
        end_utc = _as_utc(end)
        if start_utc >= end_utc:
            raise ValueError("start must be earlier than end")
        markets = self._fetch_settled_markets(
            start_utc,
            end_utc,
            only_event_tickers=only_event_tickers,
        )
        self._emit(
            "parse",
            f"grouping {len(markets)} markets",
            markets=len(markets),
        )
        settled = list(
            parse_settled_kxbtc(
                markets,
                start=start_utc,
                end=end_utc,
                on_skip=on_skip,
            )
        )
        seen = {
            market["event_ticker"]
            for market in markets
            if isinstance(market.get("event_ticker"), str)
        }
        requested = (
            only_event_tickers
            if only_event_tickers is not None
            else expected_kxbtc_event_tickers(start_utc, end_utc)
        )
        close_by_ticker = {
            kxbtc_event_ticker(close): close
            for close in _hourly_closes(start_utc, end_utc)
        }
        for event_ticker in sorted(requested):
            if event_ticker in seen:
                continue
            close_ts = close_by_ticker.get(event_ticker)
            if close_ts is None:
                continue
            settled.append(no_markets_event(event_ticker, close_ts))
        settled.sort(
            key=lambda item: (item.event.close_ts, item.event.event_ticker)
        )
        return tuple(settled)

    def _fetch_settled_markets(
        self,
        start: datetime,
        end: datetime,
        *,
        only_event_tickers: frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        by_ticker: dict[str, dict[str, Any]] = {}
        wanted_closes: list[datetime] | None = None
        if only_event_tickers is not None:
            wanted_closes = [
                close
                for close in _hourly_closes(start, end)
                if kxbtc_event_ticker(close) in only_event_tickers
            ]
            if not wanted_closes:
                self._emit("fetch", "nothing to fetch", markets=0)
                return []

        self._emit("cutoff", "fetching historical cutoff")
        cutoff = self._market_settled_cutoff()
        self._emit(
            "cutoff",
            f"cutoff {cutoff.isoformat().replace('+00:00', 'Z')}",
        )

        live_needed = (
            True
            if wanted_closes is None
            else any(close >= cutoff for close in wanted_closes)
        )
        # Live /markets: close_ts filters (not compatible with status=settled).
        # min is "after" so pass start-1 to keep closes at start; client
        # still enforces half-open [start, end).
        if live_needed:
            self._emit("live", "fetching live markets in range")
            for market in self._iter_markets(
                "/markets",
                {
                    "series_ticker": SERIES_TICKER,
                    "min_close_ts": str(_unix(start) - 1),
                    "max_close_ts": str(_unix(end)),
                },
                phase="live",
                unique=by_ticker,
            ):
                event_ticker = market.get("event_ticker")
                if (
                    only_event_tickers is not None
                    and event_ticker not in only_event_tickers
                ):
                    continue
                by_ticker[market["ticker"]] = market
        else:
            self._emit(
                "live",
                "skipped (no missing hours after cutoff)",
            )

        if wanted_closes is None:
            hist_closes = (
                list(_hourly_closes(start, min(end, cutoff)))
                if start < cutoff
                else []
            )
        else:
            hist_closes = [close for close in wanted_closes if close < cutoff]

        if hist_closes:
            total = len(hist_closes)
            self._emit(
                "historical",
                f"{total} hourly events before cutoff",
                completed=0,
                total=total,
            )
            for index, close in enumerate(hist_closes, start=1):
                event_ticker = kxbtc_event_ticker(close)
                for market in self._iter_markets(
                    "/historical/markets",
                    {"event_ticker": event_ticker},
                    phase="historical",
                    unique=by_ticker,
                    completed=index,
                    total=total,
                    detail=event_ticker,
                ):
                    by_ticker.setdefault(market["ticker"], market)
        elif wanted_closes is not None:
            self._emit(
                "historical",
                "skipped (no missing hours before cutoff)",
            )
        else:
            self._emit(
                "historical",
                "skipped (range is entirely after cutoff)",
            )

        self._emit(
            "fetch",
            f"fetched {len(by_ticker)} unique markets",
            markets=len(by_ticker),
        )
        return list(by_ticker.values())

    def _market_settled_cutoff(self) -> datetime:
        payload = self._get_json("/historical/cutoff")
        return _parse_utc(payload["market_settled_ts"])

    def market_settled_cutoff(self) -> datetime:
        """Public wrapper for ``GET /historical/cutoff`` → ``market_settled_ts``."""
        return self._market_settled_cutoff()

    def fetch_market_quote_candles(
        self,
        ticker: str,
        *,
        start_ts: int,
        end_ts: int,
        close_ts: int,
        cutoff: datetime | None = None,
    ):
        """Fetch 1m YES bid/ask closes for one ticker; route live vs historical.

        ``start_ts``/``end_ts`` are inclusive candle-end bounds. Primary route
        uses ``market_settled_ts`` vs event ``close_ts``; on 404 the other
        endpoint is tried once (boundary fallback).
        """
        from bob.db import MarketQuoteBar

        if start_ts > end_ts:
            raise ValueError("start_ts must be <= end_ts")
        settled_cutoff = cutoff if cutoff is not None else self._market_settled_cutoff()
        if close_ts < int(settled_cutoff.timestamp()):
            ordered = ("historical", "live")
        else:
            ordered = ("live", "historical")
        for kind in ordered:
            if kind == "historical":
                path = f"/historical/markets/{ticker}/candlesticks"
            else:
                path = f"/series/{SERIES_TICKER}/markets/{ticker}/candlesticks"
            payload = self._get_json(
                path,
                params={
                    "start_ts": str(start_ts),
                    "end_ts": str(end_ts),
                    "period_interval": "1",
                },
                not_found_ok=True,
            )
            if payload is None:
                continue
            if not isinstance(payload, Mapping):
                raise KalshiParseError(
                    f"market candlesticks payload is not an object for {ticker!r}"
                )
            response_ticker = payload.get("ticker")
            if response_ticker is not None and response_ticker != ticker:
                raise KalshiParseError(
                    f"candlesticks ticker mismatch: wanted {ticker!r}, "
                    f"got {response_ticker!r}"
                )
            sticks = payload.get("candlesticks")
            if not isinstance(sticks, list):
                raise KalshiParseError(
                    f"market candlesticks missing list for {ticker!r}"
                )
            bars: list[MarketQuoteBar] = []
            for row in sticks:
                if not isinstance(row, Mapping):
                    raise KalshiParseError(
                        f"candlestick is not an object for {ticker!r}: {row!r}"
                    )
                bar = parse_market_candlestick(ticker, row)
                if start_ts <= bar.end_ts <= end_ts:
                    bars.append(bar)
            return tuple(bars)
        # Both endpoints 404 → empty hour (caller pads nulls).
        return ()

    def _iter_markets(
        self,
        path: str,
        params: Mapping[str, str],
        *,
        phase: str,
        unique: dict[str, dict[str, Any]],
        completed: int | None = None,
        total: int | None = None,
        detail: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        page = 0
        while True:
            query = {**params, "limit": str(PAGE_LIMIT)}
            if cursor:
                query["cursor"] = cursor
            payload = self._get_json(path, params=query)
            page += 1
            page_markets = 0
            for market in payload.get("markets", []):
                if isinstance(market, dict):
                    page_markets += 1
                    yield market
            self._markets = len(unique)
            label = detail or path
            self._emit(
                phase,
                f"{label} page {page} (+{page_markets})",
                markets=len(unique),
                completed=completed,
                total=total,
            )
            cursor = payload.get("cursor") or None
            if not cursor:
                break

    def fetch_brti_minute_bars(
        self,
        *,
        start: datetime,
        end: datetime,
        hour_starts: Iterable[datetime] | None = None,
    ):
        """Fetch BRTI HOUR history and aggregate to 1m bars in ``[start, end)``."""
        if self._credentials is None:
            raise KalshiCredentialsError(
                "Kalshi credentials required for CF Benchmarks history"
            )
        start_utc = _as_utc(start)
        end_utc = _as_utc(end)
        if start_utc >= end_utc:
            raise ValueError("start must be earlier than end")
        hours = (
            list(hour_starts)
            if hour_starts is not None
            else _utc_hour_starts(start_utc, end_utc)
        )
        bars: list[Any] = []
        total = len(hours)
        for index, hour_start in enumerate(hours, start=1):
            hour_start = _as_utc(hour_start).replace(
                minute=0, second=0, microsecond=0
            )
            self._emit(
                "candles",
                f"BRTI {hour_start.isoformat().replace('+00:00', 'Z')}",
                completed=index,
                total=total,
            )
            payload = self._get_json(
                CF_HISTORY_VALUES_PATH,
                params={
                    "id": BRTI_INDEX_ID,
                    "timespan": "HOUR",
                    # CF requires ISO_INSTANT, not unix seconds/ms.
                    "timestamp": _cf_iso_instant(hour_start),
                },
                sign=True,
            )
            samples = parse_cf_history_samples(payload)
            bars.extend(
                aggregate_samples_to_minute_bars(
                    samples, start=start_utc, end=end_utc
                )
            )
        return tuple(bars)

    def _get_json(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        sign: bool = False,
        not_found_ok: bool = False,
    ) -> Any:
        delay = _INITIAL_429_DELAY_S
        for attempt in range(_MAX_429_ATTEMPTS):
            self._limiter.wait()
            headers: dict[str, str] | None = None
            if sign:
                if self._credentials is None:
                    raise KalshiCredentialsError(
                        "Kalshi credentials required for signed requests"
                    )
                headers = auth_headers(self._credentials, "GET", path)
            response = self._http.get(path, params=params, headers=headers)
            self._requests += 1
            if response.status_code == 429:
                self._retries_429 += 1
                self._limiter.note_429()
                if attempt == _MAX_429_ATTEMPTS - 1:
                    response.raise_for_status()
                self._emit(
                    "throttle",
                    f"429 on {path}; backing off {delay:.1f}s "
                    f"(limit ~{self._limiter.rps:.1f} req/s)",
                )
                self._sleep(delay)
                delay = min(delay * 2, _MAX_429_DELAY_S)
                continue
            if response.status_code == 401:
                raise KalshiAuthError(
                    "Kalshi authentication failed (401): bad key or signature"
                )
            if response.status_code == 403:
                raise KalshiAuthError(
                    "Kalshi request forbidden (403)"
                )
            if response.status_code == 404 and not_found_ok:
                self._limiter.note_success()
                return None
            if response.status_code == 503:
                if sign:
                    raise KalshiAuthError(
                        "Kalshi CF passthrough unavailable (503): "
                        "entitlement missing or upstream outage"
                    )
                raise KalshiAPIError(
                    f"Kalshi service unavailable (503) for {path}: "
                    f"{_response_error_detail(response)}"
                )
            if response.status_code == 400:
                raise KalshiAPIError(
                    f"Kalshi bad request (400) for {path}: "
                    f"{_response_error_detail(response)}"
                )
            response.raise_for_status()
            self._limiter.note_success()
            return response.json()
        raise AssertionError("unreachable")

    def _emit(
        self,
        phase: str,
        detail: str,
        *,
        markets: int | None = None,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        if self._on_update is None:
            return
        self._on_update(
            FetchUpdate(
                phase=phase,
                detail=detail,
                requests=self._requests,
                markets=self._markets if markets is None else markets,
                retries_429=self._retries_429,
                completed=completed,
                total=total,
            )
        )


def _utc_hour_starts(start: datetime, end: datetime) -> list[datetime]:
    start_utc = _as_utc(start)
    end_utc = _as_utc(end)
    cursor = start_utc.replace(minute=0, second=0, microsecond=0)
    hours: list[datetime] = []
    while cursor < end_utc:
        hours.append(cursor)
        cursor += timedelta(hours=1)
    return hours


def _cf_iso_instant(value: datetime) -> str:
    """CF history ``timestamp`` query value (ISO-8601 instant, UTC)."""
    return (
        _as_utc(value)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%S.000Z")
    )


def _response_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or response.reason_phrase
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            details = error.get("details")
            message = error.get("message")
            if details:
                return str(details)
            if message:
                return str(message)
        if "message" in payload:
            return str(payload["message"])
    return response.text.strip() or response.reason_phrase


def load_private_key(path: Path | str) -> RSAPrivateKey:
    key_path = Path(path)
    try:
        pem = key_path.read_bytes()
        key = serialization.load_pem_private_key(
            pem,
            password=None,
            backend=default_backend(),
        )
    except OSError as exc:
        raise KalshiCredentialsError(
            f"cannot read KALSHI_PRIVATE_KEY_PATH: {key_path}"
        ) from exc
    except (ValueError, TypeError) as exc:
        raise KalshiCredentialsError(
            f"invalid private key at KALSHI_PRIVATE_KEY_PATH: {key_path}"
        ) from exc
    if not isinstance(key, RSAPrivateKey):
        raise KalshiCredentialsError(f"not an RSA private key: {key_path}")
    return key


def sign_pss_text(private_key: RSAPrivateKey, text: str) -> str:
    signature = private_key.sign(
        text.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def signing_path(base_url: str, path: str) -> str:
    """Full Trade API path for signing (no query string)."""
    root = urlparse(base_url).path.rstrip("/")
    relative = path.split("?", 1)[0]
    if not relative.startswith("/"):
        relative = f"/{relative}"
    return f"{root}{relative}"


def auth_headers(
    credentials: KalshiCredentials,
    method: str,
    path: str,
    *,
    timestamp_ms: str | None = None,
) -> dict[str, str]:
    stamp = timestamp_ms or str(int(time.time() * 1000))
    signed = signing_path(credentials.base_url, path)
    message = f"{stamp}{method.upper()}{signed}"
    return {
        "KALSHI-ACCESS-KEY": credentials.api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": stamp,
        "KALSHI-ACCESS-SIGNATURE": sign_pss_text(
            credentials.private_key, message
        ),
    }


def load_dotenv(path: Path | str = ".env") -> None:
    """Load ``KEY=value`` pairs from ``path`` without overriding existing env."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'").strip('"')


def require_kalshi_credentials(
    *,
    env: Mapping[str, str] | None = None,
) -> KalshiCredentials:
    """Load API key id + PEM path from the environment (fail fast)."""
    source = os.environ if env is None else env
    key_id = (source.get("KALSHI_API_KEY_ID") or "").strip()
    key_path = (source.get("KALSHI_PRIVATE_KEY_PATH") or "").strip()
    base_url = (source.get("KALSHI_BASE_URL") or BASE_URL).strip().rstrip("/")
    if not key_id:
        raise KalshiCredentialsError(
            "missing KALSHI_API_KEY_ID (set it in the environment or .env)"
        )
    if not key_path:
        raise KalshiCredentialsError(
            "missing KALSHI_PRIVATE_KEY_PATH (set it in the environment or .env)"
        )
    path = Path(key_path)
    if not path.is_file():
        raise KalshiCredentialsError(
            f"KALSHI_PRIVATE_KEY_PATH is not a file: {key_path}"
        )
    return KalshiCredentials(
        api_key_id=key_id,
        private_key=load_private_key(path),
        base_url=base_url,
    )


def normalize_cf_time(raw: int | float) -> int:
    """Convert CF ``time`` to Unix seconds (ms values are >= 1e12)."""
    value = int(raw)
    if value >= 1_000_000_000_000:
        return value // 1000
    return value


def parse_cf_history_samples(payload: Any) -> list[tuple[int, Decimal]]:
    """Parse CF history/values samples as ``(unix_sec, value)`` ascending."""
    rows = _cf_sample_rows(payload)
    samples: list[tuple[int, Decimal]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise KalshiParseError(f"CF sample is not an object: {row!r}")
        if "time" not in row or "value" not in row:
            raise KalshiParseError(f"CF sample missing time/value: {row!r}")
        samples.append(
            (normalize_cf_time(row["time"]), Decimal(str(row["value"])))
        )
    samples.sort(key=lambda item: item[0])
    return samples


def _cf_sample_rows(payload: Any) -> Sequence[Any]:
    if isinstance(payload, Mapping) and "data" in payload:
        return _cf_sample_rows(payload["data"])
    if isinstance(payload, Mapping) and "payload" in payload:
        inner = payload["payload"]
        if isinstance(inner, list):
            return inner
        if isinstance(inner, Mapping) and "value" in inner:
            rows = inner["value"]
            if isinstance(rows, list):
                return rows
        raise KalshiParseError(
            f"unexpected CF payload shape: {type(inner)!r}"
        )
    if isinstance(payload, list):
        return payload
    raise KalshiParseError(f"unexpected CF response shape: {type(payload)!r}")


def _quote_close_dollars(distribution: Any, *, field: str) -> str:
    """Extract a required fixed-point dollar close from bid/ask OHLC."""
    if distribution is None:
        raise KalshiParseError(f"missing {field} distribution")
    if not isinstance(distribution, Mapping):
        raise KalshiParseError(f"{field} is not an object: {distribution!r}")
    # Live uses *_dollars; historical uses bare open/high/low/close.
    raw = distribution.get("close_dollars")
    if raw is None:
        raw = distribution.get("close")
    if raw is None:
        raise KalshiParseError(f"missing {field} close")
    text = str(raw).strip()
    if not text:
        raise KalshiParseError(f"empty {field} close")
    try:
        amount = Decimal(text)
    except InvalidOperation as exc:
        raise KalshiParseError(f"invalid {field} close: {raw!r}") from exc
    if not amount.is_finite():
        raise KalshiParseError(f"non-finite {field} close: {raw!r}")
    return format(amount, "f")


def parse_market_candlestick(ticker: str, row: Mapping[str, Any]):
    """Normalize live or historical market candlestick → ``MarketQuoteBar``."""
    from bob.db import MarketQuoteBar

    end_raw = row.get("end_period_ts")
    if end_raw is None:
        raise KalshiParseError(f"candlestick missing end_period_ts: {row!r}")
    try:
        end_ts = int(end_raw)
    except (TypeError, ValueError) as exc:
        raise KalshiParseError(f"bad end_period_ts: {end_raw!r}") from exc
    return MarketQuoteBar(
        ticker=ticker,
        end_ts=end_ts,
        yes_bid_close=_quote_close_dollars(row.get("yes_bid"), field="yes_bid"),
        yes_ask_close=_quote_close_dollars(row.get("yes_ask"), field="yes_ask"),
    )


def aggregate_samples_to_minute_bars(
    samples: Iterable[tuple[int, Decimal]],
    *,
    start: datetime,
    end: datetime,
):
    """Aggregate samples into 1m OHLC bars clipped to ``[start, end)``.

    Bar ``end_ts`` is the exclusive minute end. A bar is kept when
    ``end_ts > start_unix AND end_ts <= end_unix``.
    """
    # Local import avoids an import cycle with bob.db.
    from bob.db import MinuteBar

    start_unix = _unix(start)
    end_unix = _unix(end)
    if start_unix >= end_unix:
        return ()

    # Bucket by full UTC minute first. A mid-minute ``start`` must not drop
    # earlier ticks from that minute, or stored OHLC is wrong and later
    # backfills will skip the incomplete bar as "present".
    buckets: dict[int, list[Decimal]] = {}
    for ts, value in samples:
        minute_start = ts - (ts % 60)
        end_ts = minute_start + 60
        if start_unix < end_ts <= end_unix:
            buckets.setdefault(minute_start, []).append(value)

    bars = []
    for minute_start in sorted(buckets):
        values = buckets[minute_start]
        bars.append(
            MinuteBar(
                end_ts=minute_start + 60,
                open=format(values[0], "f"),
                high=format(max(values), "f"),
                low=format(min(values), "f"),
                close=format(values[-1], "f"),
            )
        )
    return tuple(bars)
