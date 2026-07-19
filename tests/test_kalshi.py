# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from bob.kalshi import (
    BASE_URL,
    FetchUpdate,
    KalshiClient,
    KalshiParseError,
    RateLimiter,
    kxbtc_event_ticker,
    parse_settled_kxbtc,
)

FIXTURES = Path(__file__).parent / "fixtures"
CLOSE = datetime(2099, 4, 1, 0, 0, tzinfo=timezone.utc)
CLOSE_ISO = "2099-04-01T00:00:00Z"
EVENT = "KXBTC-99APR0100"
WINNER = "KXBTC-99APR0100-B420"
LOWER = "KXBTC-99APR0100-T100"
UPPER = "KXBTC-99APR0100-T999999"
EXPIRATION = "420.69"
START = CLOSE
END = datetime(2099, 4, 2, tzinfo=timezone.utc)
# Range is after this cutoff → live only, no historical scan.
OLD_CUTOFF = {
    "market_settled_ts": "2000-01-01T00:00:00Z",
    "trades_created_ts": "2000-01-01T00:00:00Z",
    "orders_updated_ts": "2000-01-01T00:00:00Z",
}


def _load_markets() -> list[dict]:
    return json.loads((FIXTURES / "kxbtc_settled_event.json").read_text())


def _bracket(
    ticker: str,
    *,
    result: str,
    floor_strike: float | None = None,
    cap_strike: float | None = None,
) -> dict:
    row: dict = {
        "ticker": ticker,
        "event_ticker": EVENT,
        "status": "finalized",
        "result": result,
        "close_time": CLOSE_ISO,
        "expiration_value": EXPIRATION,
    }
    if floor_strike is not None:
        row["floor_strike"] = floor_strike
    if cap_strike is not None:
        row["cap_strike"] = cap_strike
    return row


def test_parse_settled_kxbtc_groups_event() -> None:
    events = parse_settled_kxbtc(_load_markets())
    assert len(events) == 1
    settled = events[0]
    assert settled.event.event_ticker == EVENT
    assert settled.event.close_ts == CLOSE
    assert settled.event.expiration_value == Decimal(EXPIRATION)
    assert len(settled.brackets) == 3
    winners = [b for b in settled.brackets if b.won]
    assert len(winners) == 1
    assert winners[0].ticker == WINNER
    assert winners[0].floor_strike == Decimal("400")
    assert winners[0].cap_strike == Decimal("499.99")


def test_parse_filters_close_time_half_open() -> None:
    markets = _load_markets()
    end = datetime(2099, 4, 1, 1, 0, tzinfo=timezone.utc)
    assert len(parse_settled_kxbtc(markets, start=CLOSE, end=end)) == 1
    assert len(parse_settled_kxbtc(markets, start=end, end=None)) == 0
    assert len(parse_settled_kxbtc(markets, start=None, end=CLOSE)) == 0


def test_parse_rejects_naive_bounds() -> None:
    markets = _load_markets()
    naive = datetime(2099, 4, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_settled_kxbtc(markets, start=naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_settled_kxbtc(markets, end=naive)


def test_parse_rejects_naive_close_time() -> None:
    markets = _load_markets()
    for market in markets:
        market["close_time"] = "2099-04-01T00:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_settled_kxbtc(markets)


def test_parse_skips_invalid_events_outside_range() -> None:
    markets = _load_markets()
    for market in markets:
        market["result"] = "no"
    assert parse_settled_kxbtc(markets, end=CLOSE) == ()


def test_parse_rejects_zero_winners() -> None:
    markets = _load_markets()
    for market in markets:
        market["result"] = "no"
    with pytest.raises(KalshiParseError, match="exactly one winner"):
        parse_settled_kxbtc(markets)


def test_parse_rejects_inconsistent_expiration() -> None:
    markets = _load_markets()
    markets[0]["expiration_value"] = "1.00"
    with pytest.raises(KalshiParseError, match="inconsistent expiration_value"):
        parse_settled_kxbtc(markets)


def test_parse_ignores_blank_expiration_values() -> None:
    markets = _load_markets()
    markets[1]["expiration_value"] = ""
    markets[2]["expiration_value"] = "  "
    events = parse_settled_kxbtc(markets)
    assert len(events) == 1
    assert events[0].event.expiration_value == Decimal(EXPIRATION)


def test_parse_on_skip_continues() -> None:
    markets = _load_markets()
    markets[0]["expiration_value"] = "1.00"
    skipped: list[str] = []
    events = parse_settled_kxbtc(
        markets,
        on_skip=lambda ticker, reason: skipped.append(ticker),
    )
    assert events == ()
    assert skipped == [EVENT]


def test_parse_rejects_non_finalized() -> None:
    markets = _load_markets()
    markets[0]["status"] = "closed"
    with pytest.raises(KalshiParseError, match="not finalized"):
        parse_settled_kxbtc(markets)


def test_kxbtc_event_ticker_uses_eastern_hour() -> None:
    close = datetime(2026, 7, 16, 0, 0, tzinfo=timezone.utc)  # 20:00 EDT
    assert kxbtc_event_ticker(close) == "KXBTC-26JUL1520"
    assert close.astimezone(ZoneInfo("America/New_York")).hour == 20


def test_fetch_date_bounds_live_and_skips_historical() -> None:
    page1 = {
        "cursor": "next",
        "markets": [
            _bracket(WINNER, result="yes", floor_strike=400, cap_strike=499.99),
        ],
    }
    page2 = {
        "cursor": "",
        "markets": [
            _bracket(LOWER, result="no", cap_strike=100),
            _bracket(UPPER, result="no", floor_strike=999999),
        ],
    }
    calls: list[tuple[str, httpx.QueryParams]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, request.url.params))
        path = request.url.path
        if path.endswith("/historical/cutoff"):
            return httpx.Response(200, json=OLD_CUTOFF)
        if path.endswith("/historical/markets"):
            return httpx.Response(500, json={"message": "should not call"})
        if path.endswith("/markets"):
            if request.url.params.get("cursor") == "next":
                return httpx.Response(200, json=page2)
            return httpx.Response(200, json=page1)
        return httpx.Response(404, json={"message": "not found"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        events = KalshiClient(http, max_rps=0).fetch_settled_kxbtc(
            start=START, end=END
        )

    assert len(events) == 1
    assert len(events[0].brackets) == 3
    live_calls = [
        (path, params)
        for path, params in calls
        if path.endswith("/markets") and not path.endswith("/historical/markets")
    ]
    assert len(live_calls) == 2
    assert live_calls[1][1].get("cursor") == "next"
    assert live_calls[0][1]["series_ticker"] == "KXBTC"
    assert "status" not in live_calls[0][1]
    assert live_calls[0][1]["min_close_ts"] == str(int(START.timestamp()) - 1)
    assert live_calls[0][1]["max_close_ts"] == str(int(END.timestamp()))
    assert live_calls[0][1]["limit"] == "1000"
    assert not any(path.endswith("/historical/markets") for path, _ in calls)


def test_fetch_historical_by_event_ticker() -> None:
    # 2024-04-01 00:00 EDT == 04:00 UTC
    start = datetime(2024, 4, 1, 4, 0, tzinfo=timezone.utc)
    end = datetime(2024, 4, 1, 5, 0, tzinfo=timezone.utc)
    event = kxbtc_event_ticker(start)
    assert event == "KXBTC-24APR0100"
    close_iso = "2024-04-01T04:00:00Z"
    hist1 = {
        "cursor": "h2",
        "markets": [
            {
                "ticker": f"{event}-B420",
                "event_ticker": event,
                "status": "finalized",
                "result": "yes",
                "close_time": close_iso,
                "expiration_value": EXPIRATION,
                "floor_strike": 400,
                "cap_strike": 499.99,
            },
        ],
    }
    hist2 = {
        "cursor": "",
        "markets": [
            {
                "ticker": f"{event}-T100",
                "event_ticker": event,
                "status": "finalized",
                "result": "no",
                "close_time": close_iso,
                "expiration_value": EXPIRATION,
                "cap_strike": 100,
            },
            {
                "ticker": f"{event}-T999999",
                "event_ticker": event,
                "status": "finalized",
                "result": "no",
                "close_time": close_iso,
                "expiration_value": EXPIRATION,
                "floor_strike": 999999,
            },
        ],
    }
    hist_calls: list[httpx.QueryParams] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/historical/cutoff"):
            return httpx.Response(
                200,
                json={
                    "market_settled_ts": "2025-01-01T00:00:00Z",
                    "trades_created_ts": "2025-01-01T00:00:00Z",
                    "orders_updated_ts": "2025-01-01T00:00:00Z",
                },
            )
        if path.endswith("/historical/markets"):
            hist_calls.append(request.url.params)
            if request.url.params.get("cursor") == "h2":
                return httpx.Response(200, json=hist2)
            return httpx.Response(200, json=hist1)
        if path.endswith("/markets"):
            return httpx.Response(200, json={"cursor": "", "markets": []})
        return httpx.Response(404, json={"message": "not found"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        events = KalshiClient(http, max_rps=0).fetch_settled_kxbtc(
            start=start, end=end
        )

    assert len(events) == 1
    assert events[0].event.event_ticker == event
    assert len(hist_calls) == 2
    assert hist_calls[0]["event_ticker"] == event
    assert "series_ticker" not in hist_calls[0]
    assert hist_calls[1].get("cursor") == "h2"


def test_fetch_retries_429() -> None:
    sleeps: list[float] = []
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/historical/cutoff"):
            return httpx.Response(200, json=OLD_CUTOFF)
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, json={"error": "too many requests"})
        return httpx.Response(
            200,
            json={
                "cursor": "",
                "markets": [
                    _bracket(
                        WINNER, result="yes", floor_strike=400, cap_strike=499.99
                    ),
                    _bracket(LOWER, result="no", cap_strike=100),
                    _bracket(UPPER, result="no", floor_strike=999999),
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        events = KalshiClient(
            http, max_rps=0, sleep=sleeps.append
        ).fetch_settled_kxbtc(start=START, end=END)
    assert len(events) == 1
    assert attempts["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_fetch_requires_ordered_bounds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OLD_CUTOFF)

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        with pytest.raises(ValueError, match="earlier"):
            KalshiClient(http, max_rps=0).fetch_settled_kxbtc(
                start=END, end=START
            )


def test_fetch_raises_for_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/historical/cutoff"):
            return httpx.Response(200, json=OLD_CUTOFF)
        return httpx.Response(500, json={"message": "boom"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        with pytest.raises(httpx.HTTPStatusError):
            KalshiClient(http, max_rps=0).fetch_settled_kxbtc(
                start=START, end=END
            )


def test_rate_limiter_paces_and_slows_on_429() -> None:
    now = {"t": 0.0}
    sleeps: list[float] = []

    def monotonic() -> float:
        return now["t"]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now["t"] += seconds

    limiter = RateLimiter(10.0, sleep=sleep, monotonic=monotonic)
    limiter.wait()
    now["t"] += 0.05
    limiter.wait()
    assert sleeps == pytest.approx([0.05])
    limiter.note_429()
    assert limiter.rps == 5.0
    limiter.note_success()
    assert limiter.rps == pytest.approx(6.25)


def test_fetch_emits_progress_updates() -> None:
    updates: list[FetchUpdate] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/historical/cutoff"):
            return httpx.Response(200, json=OLD_CUTOFF)
        return httpx.Response(
            200,
            json={
                "cursor": "",
                "markets": [
                    _bracket(
                        WINNER, result="yes", floor_strike=400, cap_strike=499.99
                    ),
                    _bracket(LOWER, result="no", cap_strike=100),
                    _bracket(UPPER, result="no", floor_strike=999999),
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        KalshiClient(
            http, max_rps=0, on_update=updates.append
        ).fetch_settled_kxbtc(start=START, end=END)

    phases = [update.phase for update in updates]
    assert "cutoff" in phases
    assert "live" in phases
    assert "historical" in phases
    assert "parse" in phases
    assert any("skipped" in update.detail for update in updates)
