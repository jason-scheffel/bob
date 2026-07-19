# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone
from decimal import Decimal

import httpx
import pytest

from bob.db import (
    MARKET_HOUR_MINUTES,
    MarketQuoteBar,
    connect,
    expected_market_minute_ends,
    initialize_schema,
    normalize_market_quote_hour,
    store_market_candles,
    store_settled_events,
    ticker_has_complete_market_hour,
)
from bob.kalshi import (
    BASE_URL,
    Bracket,
    Event,
    KalshiClient,
    SettledEvent,
    parse_market_candlestick,
)
from bob.market_candles import run_backfill_market_candles

CLOSE = datetime(2099, 4, 1, 0, 0, tzinfo=timezone.utc)
CLOSE_TS = int(CLOSE.timestamp())
TICKER = "KXBTC-99APR0100-B420"
EVENT = "KXBTC-99APR0100"


def _settled() -> SettledEvent:
    return SettledEvent(
        event=Event(
            event_ticker=EVENT,
            close_ts=CLOSE,
            status="complete",
            expiration_value=Decimal("420"),
        ),
        brackets=(
            Bracket(
                ticker=TICKER,
                event_ticker=EVENT,
                floor_strike=Decimal("400"),
                cap_strike=Decimal("499.99"),
                won=True,
            ),
        ),
    )


def test_expected_market_minute_ends_are_sixty_inclusive() -> None:
    ends = expected_market_minute_ends(CLOSE_TS)
    assert len(ends) == MARKET_HOUR_MINUTES
    assert ends[0] == CLOSE_TS - 3540
    assert ends[-1] == CLOSE_TS
    assert ends == list(range(CLOSE_TS - 3540, CLOSE_TS + 1, 60))


def test_normalize_pads_sparse_to_sixty_nulls() -> None:
    sparse = (
        MarketQuoteBar(
            ticker=TICKER,
            end_ts=CLOSE_TS - 60,
            yes_bid_close="0.40",
            yes_ask_close="0.42",
        ),
    )
    hour = normalize_market_quote_hour(TICKER, CLOSE_TS, sparse)
    assert len(hour) == 60
    filled = [bar for bar in hour if bar.yes_bid_close is not None]
    assert len(filled) == 1
    assert filled[0].end_ts == CLOSE_TS - 60
    assert hour[-1].yes_bid_close is None


def test_parse_live_and_historical_quote_fields() -> None:
    live = parse_market_candlestick(
        TICKER,
        {
            "end_period_ts": CLOSE_TS,
            "yes_bid": {"close_dollars": "0.4100"},
            "yes_ask": {"close_dollars": "0.4300"},
        },
    )
    hist = parse_market_candlestick(
        TICKER,
        {
            "end_period_ts": CLOSE_TS,
            "yes_bid": {"close": "0.4100"},
            "yes_ask": {"close": "0.4300"},
        },
    )
    assert live.yes_bid_close == "0.4100"
    assert live.yes_ask_close == "0.4300"
    assert hist.yes_bid_close == "0.4100"
    assert hist.yes_ask_close == "0.4300"


def test_fetch_routes_historical_before_cutoff() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/trade-api/v2/historical/cutoff":
            return httpx.Response(
                200,
                json={
                    "market_settled_ts": "2100-01-01T00:00:00Z",
                    "trades_created_ts": "2100-01-01T00:00:00Z",
                    "orders_updated_ts": "2100-01-01T00:00:00Z",
                },
            )
        if "/historical/markets/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "ticker": TICKER,
                    "candlesticks": [
                        {
                            "end_period_ts": CLOSE_TS,
                            "yes_bid": {"close": "0.50"},
                            "yes_ask": {"close": "0.55"},
                        }
                    ],
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport) as http:
        client = KalshiClient(http, max_rps=0)
        bars = client.fetch_market_quote_candles(
            TICKER,
            start_ts=CLOSE_TS - 3540,
            end_ts=CLOSE_TS,
            close_ts=CLOSE_TS,
        )
    assert len(bars) == 1
    assert bars[0].yes_ask_close == "0.55"
    assert any("/historical/markets/" in path for path in paths)
    assert not any("/series/KXBTC/markets/" in path for path in paths)


def test_fetch_falls_back_on_404() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/historical/cutoff"):
            return httpx.Response(
                200,
                json={
                    "market_settled_ts": "2100-01-01T00:00:00Z",
                    "trades_created_ts": "2100-01-01T00:00:00Z",
                    "orders_updated_ts": "2100-01-01T00:00:00Z",
                },
            )
        if "/historical/markets/" in request.url.path:
            return httpx.Response(404)
        if "/series/KXBTC/markets/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "ticker": TICKER,
                    "candlesticks": [
                        {
                            "end_period_ts": CLOSE_TS,
                            "yes_bid": {"close_dollars": "0.20"},
                            "yes_ask": {"close_dollars": "0.22"},
                        }
                    ],
                },
            )
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport) as http:
        client = KalshiClient(http, max_rps=0)
        bars = client.fetch_market_quote_candles(
            TICKER,
            start_ts=CLOSE_TS - 3540,
            end_ts=CLOSE_TS,
            close_ts=CLOSE_TS,
        )
    assert bars[0].yes_bid_close == "0.20"
    assert any("/historical/markets/" in path for path in paths)
    assert any("/series/KXBTC/markets/" in path for path in paths)


def test_backfill_writes_sixty_and_skips_on_restart() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_settled_events(connection, [_settled()])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/historical/cutoff"):
            return httpx.Response(
                200,
                json={
                    "market_settled_ts": "2000-01-01T00:00:00Z",
                    "trades_created_ts": "2000-01-01T00:00:00Z",
                    "orders_updated_ts": "2000-01-01T00:00:00Z",
                },
            )
        if "/candlesticks" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "ticker": TICKER,
                    "candlesticks": [
                        {
                            "end_period_ts": CLOSE_TS - 120,
                            "yes_bid": {"close_dollars": "0.33"},
                            "yes_ask": {"close_dollars": "0.34"},
                        }
                    ],
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    start = CLOSE
    end = datetime(2099, 4, 1, 1, tzinfo=timezone.utc)
    with httpx.Client(base_url=BASE_URL, transport=transport) as http:
        client = KalshiClient(http, max_rps=0)
        first = run_backfill_market_candles(connection, client, start, end)
        assert first.fetched == 1
        assert first.written == 60
        assert ticker_has_complete_market_hour(connection, TICKER, CLOSE_TS)
        second = run_backfill_market_candles(connection, client, start, end)
        assert second.fetched == 0
        assert second.skipped == 1
    count = connection.execute(
        "SELECT COUNT(*) FROM market_candles WHERE ticker = ?",
        (TICKER,),
    ).fetchone()[0]
    assert count == 60
    connection.close()


def test_store_null_placeholders_count_as_complete() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    hour = normalize_market_quote_hour(TICKER, CLOSE_TS, ())
    store_market_candles(connection, hour)
    assert ticker_has_complete_market_hour(connection, TICKER, CLOSE_TS)
    connection.close()


def test_malformed_candlestick_does_not_mark_complete() -> None:
    from bob.kalshi import KalshiParseError

    connection = connect(":memory:")
    initialize_schema(connection)
    store_settled_events(connection, [_settled()])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/historical/cutoff"):
            return httpx.Response(
                200,
                json={
                    "market_settled_ts": "2000-01-01T00:00:00Z",
                    "trades_created_ts": "2000-01-01T00:00:00Z",
                    "orders_updated_ts": "2000-01-01T00:00:00Z",
                },
            )
        if "/candlesticks" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "ticker": TICKER,
                    "candlesticks": [
                        {
                            "end_period_ts": CLOSE_TS,
                            "yes_bid": {"open_dollars": "0.1"},
                            "yes_ask": {"close_dollars": "0.2"},
                        }
                    ],
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    start = CLOSE
    end = datetime(2099, 4, 1, 1, tzinfo=timezone.utc)
    with httpx.Client(base_url=BASE_URL, transport=transport) as http:
        client = KalshiClient(http, max_rps=0)
        counts = run_backfill_market_candles(connection, client, start, end)
    assert counts.errors == 1
    assert counts.written == 0
    assert not ticker_has_complete_market_hour(connection, TICKER, CLOSE_TS)
    with pytest.raises(KalshiParseError):
        parse_market_candlestick(
            TICKER,
            {
                "end_period_ts": CLOSE_TS,
                "yes_bid": {"open_dollars": "0.1"},
                "yes_ask": {"close_dollars": "0.2"},
            },
        )
    connection.close()


def test_non_404_error_does_not_fall_back() -> None:
    from bob.kalshi import KalshiAPIError

    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/historical/cutoff"):
            return httpx.Response(
                200,
                json={
                    "market_settled_ts": "2100-01-01T00:00:00Z",
                    "trades_created_ts": "2100-01-01T00:00:00Z",
                    "orders_updated_ts": "2100-01-01T00:00:00Z",
                },
            )
        if "/historical/markets/" in request.url.path:
            return httpx.Response(500, json={"message": "boom"})
        return httpx.Response(200, json={"ticker": TICKER, "candlesticks": []})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport) as http:
        client = KalshiClient(http, max_rps=0)
        with pytest.raises((KalshiAPIError, httpx.HTTPStatusError)):
            client.fetch_market_quote_candles(
                TICKER,
                start_ts=CLOSE_TS - 3540,
                end_ts=CLOSE_TS,
                close_ts=CLOSE_TS,
            )
    assert any("/historical/markets/" in path for path in paths)
    assert not any("/series/KXBTC/markets/" in path for path in paths)
