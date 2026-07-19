# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from bob.kalshi import (
    BASE_URL,
    KalshiClient,
    KalshiParseError,
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


def test_parse_rejects_non_finalized() -> None:
    markets = _load_markets()
    markets[0]["status"] = "closed"
    with pytest.raises(KalshiParseError, match="not finalized"):
        parse_settled_kxbtc(markets)


def test_fetch_paginates_and_merges_sources() -> None:
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
    historical = {
        "cursor": "",
        "markets": [
            # Duplicate ticker from live source — should dedupe, keep first.
            _bracket(WINNER, result="yes", floor_strike=400, cap_strike=499.99),
        ],
    }

    calls: list[tuple[str, httpx.QueryParams]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, request.url.params))
        path = request.url.path
        if path.endswith("/markets") and not path.endswith("/historical/markets"):
            if request.url.params.get("cursor") == "next":
                return httpx.Response(200, json=page2)
            return httpx.Response(200, json=page1)
        if path.endswith("/historical/markets"):
            return httpx.Response(200, json=historical)
        return httpx.Response(404, json={"message": "not found"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        events = KalshiClient(http).fetch_settled_kxbtc()

    assert len(events) == 1
    assert len(events[0].brackets) == 3
    live_calls = [
        (path, params)
        for path, params in calls
        if path.endswith("/markets") and not path.endswith("/historical/markets")
    ]
    hist_calls = [
        path for path, _ in calls if path.endswith("/historical/markets")
    ]
    assert len(live_calls) == 2
    assert live_calls[1][1].get("cursor") == "next"
    assert len(hist_calls) == 1
    assert live_calls[0][1]["series_ticker"] == "KXBTC"
    assert live_calls[0][1]["status"] == "settled"
    assert live_calls[0][1]["limit"] == "1000"


def test_fetch_paginates_historical() -> None:
    live = {"cursor": "", "markets": []}
    hist1 = {
        "cursor": "h2",
        "markets": [
            _bracket(WINNER, result="yes", floor_strike=400, cap_strike=499.99),
        ],
    }
    hist2 = {
        "cursor": "",
        "markets": [
            _bracket(LOWER, result="no", cap_strike=100),
            _bracket(UPPER, result="no", floor_strike=999999),
        ],
    }
    hist_cursors: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/historical/markets"):
            hist_cursors.append(request.url.params.get("cursor"))
            if request.url.params.get("cursor") == "h2":
                return httpx.Response(200, json=hist2)
            return httpx.Response(200, json=hist1)
        return httpx.Response(200, json=live)

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        events = KalshiClient(http).fetch_settled_kxbtc()
    assert len(events) == 1
    assert len(events[0].brackets) == 3
    assert hist_cursors == [None, "h2"]


def test_fetch_raises_for_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "boom"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        with pytest.raises(httpx.HTTPStatusError):
            KalshiClient(http).fetch_settled_kxbtc()
