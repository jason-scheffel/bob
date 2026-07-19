# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import typer
from typer.testing import CliRunner

from bob.cli import app, parse_iso_datetime, run_backfill
from bob.db import connect, initialize_schema
from bob.kalshi import BASE_URL, KalshiClient

runner = CliRunner()


def test_parse_iso_datetime_requires_timezone() -> None:
    assert parse_iso_datetime("2099-04-01T00:00:00Z") == datetime(
        2099, 4, 1, 0, 0, tzinfo=timezone.utc
    )
    with pytest.raises(typer.BadParameter, match="timezone"):
        parse_iso_datetime("2099-04-01T00:00:00")


def test_run_backfill_half_open(tmp_path: Path) -> None:
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
        if request.url.path.endswith("/historical/markets"):
            return httpx.Response(200, json={"cursor": "", "markets": []})
        return httpx.Response(200, json=payload)

    db_path = tmp_path / "test.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=BASE_URL, transport=transport, timeout=5.0) as http:
        counts = run_backfill(
            connection,
            KalshiClient(http),
            datetime(2099, 4, 1, tzinfo=timezone.utc),
            datetime(2099, 4, 2, tzinfo=timezone.utc),
        )
    assert counts.events == 1
    assert counts.brackets == 2
    tickers = {
        row[0]
        for row in connection.execute("SELECT event_ticker FROM events")
    }
    assert tickers == {"KXBTC-99APR0100"}
    connection.close()


def test_cli_backfill_rejects_bad_range(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_cli_backfill_mocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
        if request.url.path.endswith("/historical/markets"):
            return httpx.Response(200, json={"cursor": "", "markets": []})
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs = {**kwargs, "transport": transport}
        return real_client(*args, **kwargs)

    monkeypatch.setattr("bob.cli.httpx.Client", fake_client)
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
        ],
    )
    assert result.exit_code == 0
    assert "stored 1 events, 2 brackets" in result.output
