# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import typer
from typer.testing import CliRunner

from bob.cli import (
    app,
    format_eta,
    iter_utc_day_chunks,
    parse_iso_datetime,
    run_backfill,
)
from bob.db import connect, initialize_schema, store_settled_events
from bob.kalshi import (
    BASE_URL,
    DEFAULT_MAX_RPS,
    STATUS_COMPLETE,
    STATUS_NO_MARKETS,
    Bracket,
    Event,
    KalshiClient,
    SettledEvent,
    expected_kxbtc_event_tickers,
    kxbtc_event_ticker,
)
from helpers import cf_empty_response, seed_complete_candle_hours

runner = CliRunner()
FUTURE = datetime(2100, 1, 1, tzinfo=timezone.utc)


def test_parse_iso_datetime_requires_timezone() -> None:
    assert parse_iso_datetime("2099-04-01T00:00:00Z") == datetime(
        2099, 4, 1, 0, 0, tzinfo=timezone.utc
    )
    with pytest.raises(typer.BadParameter, match="timezone"):
        parse_iso_datetime("2099-04-01T00:00:00")


def test_default_max_rps_is_three() -> None:
    assert DEFAULT_MAX_RPS == 3.0


def test_format_eta() -> None:
    assert format_eta(45) == "45s"
    assert format_eta(90) == "1m30s"
    assert format_eta(3661) == "1h01m"


def test_iter_utc_day_chunks() -> None:
    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    end = datetime(2026, 6, 3, 12, tzinfo=timezone.utc)
    chunks = list(iter_utc_day_chunks(start, end))
    assert chunks == [
        (
            datetime(2026, 6, 1, tzinfo=timezone.utc),
            datetime(2026, 6, 2, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 6, 2, tzinfo=timezone.utc),
            datetime(2026, 6, 3, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 6, 3, tzinfo=timezone.utc),
            datetime(2026, 6, 3, 12, tzinfo=timezone.utc),
        ),
    ]


def test_run_backfill_half_open(tmp_path: Path, kalshi_credentials) -> None:
    payload = {
        "cursor": "",
        "markets": [
            {
                "ticker": "KXBTC-99APR0100-B420",
                "event_ticker": "KXBTC-99APR0100",
                "status": "finalized",
                "result": "yes",
                "close_time": "2099-04-01T00:00:00Z",
                "expiration_value": "420.69",
                "floor_strike": 400,
                "cap_strike": 499.99,
            },
            {
                "ticker": "KXBTC-99APR0100-T100",
                "event_ticker": "KXBTC-99APR0100",
                "status": "finalized",
                "result": "no",
                "close_time": "2099-04-01T00:00:00Z",
                "expiration_value": "420.69",
                "cap_strike": 100,
            },
            {
                "ticker": "KXBTC-99APR0200-B420",
                "event_ticker": "KXBTC-99APR0200",
                "status": "finalized",
                "result": "yes",
                "close_time": "2099-04-02T00:00:00Z",
                "expiration_value": "421.00",
                "floor_strike": 400,
                "cap_strike": 499.99,
            },
            {
                "ticker": "KXBTC-99APR0200-T100",
                "event_ticker": "KXBTC-99APR0200",
                "status": "finalized",
                "result": "no",
                "close_time": "2099-04-02T00:00:00Z",
                "expiration_value": "421.00",
                "cap_strike": 100,
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if "cfbenchmarks" in request.url.path:
            return httpx.Response(200, json=cf_empty_response())
        if request.url.path.endswith("/historical/cutoff"):
            return httpx.Response(
                200,
                json={
                    "market_settled_ts": "2000-01-01T00:00:00Z",
                    "trades_created_ts": "2000-01-01T00:00:00Z",
                    "orders_updated_ts": "2000-01-01T00:00:00Z",
                },
            )
        if request.url.path.endswith("/historical/markets"):
            return httpx.Response(500, json={"message": "should not call"})
        return httpx.Response(200, json=payload)

    db_path = tmp_path / "test.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    transport = httpx.MockTransport(handler)
    start = datetime(2099, 4, 1, tzinfo=timezone.utc)
    end = datetime(2099, 4, 2, tzinfo=timezone.utc)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        counts = run_backfill(
            connection,
            KalshiClient(http, credentials=kalshi_credentials, max_rps=0),
            start,
            end,
            now=FUTURE,
        )
    expected = expected_kxbtc_event_tickers(start, end)
    assert counts.events == len(expected)
    assert counts.brackets == 2
    rows = {
        row[0]: row[1]
        for row in connection.execute("SELECT event_ticker, status FROM events")
    }
    assert set(rows) == expected
    assert rows["KXBTC-99APR0100"] == STATUS_COMPLETE
    assert sum(1 for status in rows.values() if status == STATUS_NO_MARKETS) == (
        len(expected) - 1
    )
    connection.close()


def _seed_event(connection, close: datetime) -> str:
    ticker = kxbtc_event_ticker(close)
    store_settled_events(
        connection,
        [
            SettledEvent(
                event=Event(
                    event_ticker=ticker,
                    close_ts=close,
                    status=STATUS_COMPLETE,
                    expiration_value=Decimal("100"),
                ),
                brackets=(
                    Bracket(
                        ticker=f"{ticker}-B100",
                        event_ticker=ticker,
                        floor_strike=Decimal("100"),
                        cap_strike=Decimal("100.99"),
                        won=True,
                    ),
                ),
            )
        ],
    )
    return ticker


def test_run_backfill_skips_full_day_already_stored(
    tmp_path: Path, kalshi_credentials
) -> None:
    start = datetime(2024, 4, 1, 4, 0, tzinfo=timezone.utc)
    end = datetime(2024, 4, 2, 4, 0, tzinfo=timezone.utc)
    connection = connect(tmp_path / "full.sqlite")
    initialize_schema(connection)
    for ticker_close in sorted(
        (
            # Reconstruct closes from expected tickers via hour walk.
            start + timedelta(hours=offset)
            for offset in range(24)
        ),
        key=lambda value: value.timestamp(),
    ):
        # Only seed hours that are expected KXBTC closes in range.
        if kxbtc_event_ticker(ticker_close) in expected_kxbtc_event_tickers(
            start, end
        ):
            _seed_event(connection, ticker_close)
    seed_complete_candle_hours(connection, start, end)

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500, json={"message": "should not call"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        counts = run_backfill(
            connection,
            KalshiClient(http, credentials=kalshi_credentials, max_rps=0),
            start,
            end,
            now=FUTURE,
        )
    assert counts.events == 0
    assert counts.candles == 0
    assert calls == []
    connection.close()


def test_run_backfill_fetches_only_missing_hour(
    tmp_path: Path, kalshi_credentials
) -> None:
    start = datetime(2024, 4, 1, 4, 0, tzinfo=timezone.utc)
    end = datetime(2024, 4, 2, 4, 0, tzinfo=timezone.utc)
    connection = connect(tmp_path / "gap.sqlite")
    initialize_schema(connection)
    expected = expected_kxbtc_event_tickers(start, end)
    closes = [
        start + timedelta(hours=offset)
        for offset in range(24)
        if kxbtc_event_ticker(start + timedelta(hours=offset)) in expected
    ]
    assert len(closes) == 24
    missing_close = closes[7]
    missing_ticker = kxbtc_event_ticker(missing_close)
    for close in closes:
        if close != missing_close:
            _seed_event(connection, close)

    hist_tickers: list[str] = []

    seed_complete_candle_hours(connection, start, end)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "cfbenchmarks" in path:
            return httpx.Response(200, json=cf_empty_response())
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
            hist_tickers.append(request.url.params["event_ticker"])
            return httpx.Response(
                200,
                json={
                    "cursor": "",
                    "markets": [
                        {
                            "ticker": f"{missing_ticker}-B100",
                            "event_ticker": missing_ticker,
                            "status": "finalized",
                            "result": "yes",
                            "close_time": missing_close.isoformat().replace(
                                "+00:00", "Z"
                            ),
                            "expiration_value": "100",
                            "floor_strike": 100,
                            "cap_strike": 100.99,
                        },
                    ],
                },
            )
        if path.endswith("/markets"):
            return httpx.Response(500, json={"message": "live should skip"})
        return httpx.Response(404, json={"message": "not found"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        counts = run_backfill(
            connection,
            KalshiClient(http, credentials=kalshi_credentials, max_rps=0),
            start,
            end,
            now=FUTURE,
        )
    assert hist_tickers == [missing_ticker]
    assert counts.events == 1
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM events WHERE event_ticker = ?",
            (missing_ticker,),
        ).fetchone()[0]
        == 1
    )
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 24
    connection.close()


def test_run_backfill_force_refetches(
    tmp_path: Path, kalshi_credentials
) -> None:
    start = datetime(2099, 4, 1, tzinfo=timezone.utc)
    end = datetime(2099, 4, 1, 1, tzinfo=timezone.utc)
    connection = connect(tmp_path / "force.sqlite")
    initialize_schema(connection)
    close = datetime(2099, 4, 1, 0, 0, tzinfo=timezone.utc)
    _seed_event(connection, close)
    live_calls = 0

    seed_complete_candle_hours(connection, start, end)
    cf_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal live_calls, cf_calls
        path = request.url.path
        if "cfbenchmarks" in path:
            cf_calls += 1
            return httpx.Response(200, json=cf_empty_response())
        if path.endswith("/historical/cutoff"):
            return httpx.Response(
                200,
                json={
                    "market_settled_ts": "2000-01-01T00:00:00Z",
                    "trades_created_ts": "2000-01-01T00:00:00Z",
                    "orders_updated_ts": "2000-01-01T00:00:00Z",
                },
            )
        if path.endswith("/historical/markets"):
            return httpx.Response(500, json={"message": "should not call"})
        if path.endswith("/markets"):
            live_calls += 1
            ticker = kxbtc_event_ticker(close)
            return httpx.Response(
                200,
                json={
                    "cursor": "",
                    "markets": [
                        {
                            "ticker": f"{ticker}-B100",
                            "event_ticker": ticker,
                            "status": "finalized",
                            "result": "yes",
                            "close_time": "2099-04-01T00:00:00Z",
                            "expiration_value": "100",
                            "floor_strike": 100,
                            "cap_strike": 100.99,
                        },
                    ],
                },
            )
        return httpx.Response(404, json={"message": "not found"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        without_force = run_backfill(
            connection,
            KalshiClient(http, credentials=kalshi_credentials, max_rps=0),
            start,
            end,
            force=False,
            now=FUTURE,
        )
        assert without_force.events == 0
        assert live_calls == 0
        assert cf_calls == 0
        with_force = run_backfill(
            connection,
            KalshiClient(http, credentials=kalshi_credentials, max_rps=0),
            start,
            end,
            force=True,
            now=FUTURE,
        )
    assert with_force.events == 1
    assert live_calls == 1
    assert cf_calls == 1
    connection.close()


def test_cli_backfill_rejects_bad_range(
    monkeypatch: pytest.MonkeyPatch, kalshi_credentials
) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    result = runner.invoke(
        app,
        [
            "backfill",
            "--start",
            "2099-04-02T00:00:00Z",
            "--end",
            "2099-04-01T00:00:00Z",
        ],
    )
    assert result.exit_code == 2
    assert "--start must be earlier than --end" in result.output


def test_cli_backfill_missing_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.setattr("bob.cli.load_dotenv", lambda: None)
    result = runner.invoke(
        app,
        [
            "backfill",
            "--start",
            "2099-04-01T00:00:00Z",
            "--end",
            "2099-04-02T00:00:00Z",
        ],
    )
    assert result.exit_code == 2
    assert "KALSHI_API_KEY_ID" in result.output


def test_cli_backfill_mocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kalshi_credentials
) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)

    payload = {
        "cursor": "",
        "markets": [
            {
                "ticker": "KXBTC-99APR0100-B420",
                "event_ticker": "KXBTC-99APR0100",
                "status": "finalized",
                "result": "yes",
                "close_time": "2099-04-01T00:00:00Z",
                "expiration_value": "420.69",
                "floor_strike": 400,
                "cap_strike": 499.99,
            },
            {
                "ticker": "KXBTC-99APR0100-T100",
                "event_ticker": "KXBTC-99APR0100",
                "status": "finalized",
                "result": "no",
                "close_time": "2099-04-01T00:00:00Z",
                "expiration_value": "420.69",
                "cap_strike": 100,
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if "cfbenchmarks" in request.url.path:
            return httpx.Response(200, json=cf_empty_response())
        if request.url.path.endswith("/historical/cutoff"):
            return httpx.Response(
                200,
                json={
                    "market_settled_ts": "2000-01-01T00:00:00Z",
                    "trades_created_ts": "2000-01-01T00:00:00Z",
                    "orders_updated_ts": "2000-01-01T00:00:00Z",
                },
            )
        if request.url.path.endswith("/historical/markets"):
            return httpx.Response(500, json={"message": "should not call"})
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs = {**kwargs, "transport": transport}
        return real_client(*args, **kwargs)

    monkeypatch.setattr("bob.cli.httpx.Client", fake_client)
    original_run = run_backfill

    def run_with_future(*args, **kwargs):
        kwargs.setdefault("now", FUTURE)
        return original_run(*args, **kwargs)

    monkeypatch.setattr("bob.cli.run_backfill", run_with_future)
    db_path = tmp_path / "cli.sqlite"
    result = runner.invoke(
        app,
        [
            "backfill",
            "--start",
            "2099-04-01T00:00:00Z",
            "--end",
            "2099-04-02T00:00:00Z",
            "--db",
            str(db_path),
            "--rps",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "[1/1] 2099-04-01" in result.output
    expected = len(
        expected_kxbtc_event_tickers(
            datetime(2099, 4, 1, tzinfo=timezone.utc),
            datetime(2099, 4, 2, tzinfo=timezone.utc),
        )
    )
    assert f"stored {expected} events, 2 brackets, 0 candles" in result.output
    assert "ETA 0s" in result.output
    assert (
        f"done  {expected} events, 2 brackets, 0 candles" in result.output
    )


def test_cli_backfill_collapses_skip_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kalshi_credentials
) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    start = datetime(2024, 4, 1, 4, 0, tzinfo=timezone.utc)
    end = datetime(2024, 4, 2, 4, 0, tzinfo=timezone.utc)
    db_path = tmp_path / "skip.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    for offset in range(24):
        close = start + timedelta(hours=offset)
        if kxbtc_event_ticker(close) in expected_kxbtc_event_tickers(start, end):
            _seed_event(connection, close)
    seed_complete_candle_hours(connection, start, end)
    connection.close()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "should not call"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs = {**kwargs, "transport": transport}
        return real_client(*args, **kwargs)

    monkeypatch.setattr("bob.cli.httpx.Client", fake_client)
    original_run = run_backfill

    def run_with_future(*args, **kwargs):
        kwargs.setdefault("now", FUTURE)
        return original_run(*args, **kwargs)

    monkeypatch.setattr("bob.cli.run_backfill", run_with_future)
    result = runner.invoke(
        app,
        [
            "backfill",
            "--start",
            "2024-04-01T04:00:00Z",
            "--end",
            "2024-04-02T04:00:00Z",
            "--db",
            str(db_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "skipped 2 days (24 hours already stored)" in result.output
    assert "done  0 events, 0 brackets, 0 candles" in result.output
    assert "[1/" not in result.output
