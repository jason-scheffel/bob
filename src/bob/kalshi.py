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
# Basic Read is 200 tokens/s at ~10 tokens/GET ≈ 20 req/s. Default stays well under.
DEFAULT_MAX_RPS = 5.0

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
    (floor 1/s) until successes climb it back toward ``max_rps``.
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
        self._rps = max(1.0, self._rps / 2.0)

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
    """Public Kalshi market client.

    Pass an ``httpx.Client`` with ``base_url=BASE_URL`` and an explicit
    ``timeout`` (auth headers can be added later on the same client).
    """

    def __init__(
        self,
        http: httpx.Client,
        *,
        max_rps: float = DEFAULT_MAX_RPS,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        on_update: Callable[[FetchUpdate], None] | None = None,
    ) -> None:
        self._http = http
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

    def _get_json(
        self, path: str, *, params: Mapping[str, str] | None = None
    ) -> Any:
        delay = _INITIAL_429_DELAY_S
        for attempt in range(_MAX_429_ATTEMPTS):
            self._limiter.wait()
            response = self._http.get(path, params=params)
            self._requests += 1
            if response.status_code != 429:
                response.raise_for_status()
                self._limiter.note_success()
                return response.json()
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
