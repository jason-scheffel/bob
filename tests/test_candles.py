# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import base64
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

from bob.kalshi import (
    BASE_URL,
    KalshiAPIError,
    KalshiAuthError,
    KalshiClient,
    KalshiCredentialsError,
    aggregate_samples_to_minute_bars,
    auth_headers,
    normalize_cf_time,
    parse_cf_history_samples,
    require_kalshi_credentials,
    sign_pss_text,
    signing_path,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_cf_time_pins_milliseconds() -> None:
    payload = json.loads((FIXTURES / "cf_brti_hour.json").read_text())
    samples = parse_cf_history_samples(payload)
    assert samples
    raw = payload["data"]["payload"][0]["time"]
    assert raw >= 1_000_000_000_000
    assert normalize_cf_time(raw) == raw // 1000
    assert samples[0][0] == raw // 1000


def test_sign_pss_and_auth_headers(kalshi_credentials) -> None:
    path = "/cfbenchmarks/history/values"
    signed = signing_path(kalshi_credentials.base_url, path)
    assert signed == "/trade-api/v2/cfbenchmarks/history/values"
    stamp = "1700000000000"
    headers = auth_headers(
        kalshi_credentials, "GET", path, timestamp_ms=stamp
    )
    assert headers["KALSHI-ACCESS-KEY"] == "test-api-key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == stamp
    message = f"{stamp}GET{signed}".encode()
    kalshi_credentials.private_key.public_key().verify(
        base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    assert sign_pss_text(kalshi_credentials.private_key, "abc")


def test_require_kalshi_credentials_fail_fast(
    monkeypatch: pytest.MonkeyPatch, rsa_pem: Path
) -> None:
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    with pytest.raises(KalshiCredentialsError, match="KALSHI_API_KEY_ID"):
        require_kalshi_credentials(env={})
    with pytest.raises(KalshiCredentialsError, match="KALSHI_PRIVATE_KEY_PATH"):
        require_kalshi_credentials(env={"KALSHI_API_KEY_ID": "x"})
    with pytest.raises(KalshiCredentialsError, match="not a file"):
        require_kalshi_credentials(
            env={
                "KALSHI_API_KEY_ID": "x",
                "KALSHI_PRIVATE_KEY_PATH": "/no/such/key.pem",
            }
        )
    creds = require_kalshi_credentials(
        env={
            "KALSHI_API_KEY_ID": "x",
            "KALSHI_PRIVATE_KEY_PATH": str(rsa_pem),
        }
    )
    assert creds.api_key_id == "x"
    assert creds.base_url == BASE_URL


def test_aggregate_samples_to_minute_bars_boundaries() -> None:
    start = datetime(2024, 7, 18, 12, 0, tzinfo=timezone.utc)
    end = datetime(2024, 7, 18, 13, 0, tzinfo=timezone.utc)
    start_ts = int(start.timestamp())
    samples = [
        (start_ts, Decimal("10")),
        (start_ts + 10, Decimal("12")),
        (start_ts + 20, Decimal("9")),
        (start_ts + 60, Decimal("11")),
        (start_ts + 3599, Decimal("15")),
        (start_ts + 3600, Decimal("99")),  # excluded (end exclusive)
    ]
    bars = aggregate_samples_to_minute_bars(samples, start=start, end=end)
    by_end = {bar.end_ts: bar for bar in bars}
    assert by_end[start_ts + 60].open == "10"
    assert by_end[start_ts + 60].high == "12"
    assert by_end[start_ts + 60].low == "9"
    assert by_end[start_ts + 60].close == "9"
    assert by_end[start_ts + 120].open == "11"
    assert by_end[start_ts + 3600].close == "15"
    assert start_ts + 3660 not in by_end


def test_aggregate_keeps_full_minute_when_start_mid_minute() -> None:
    minute = datetime(2024, 7, 18, 12, 30, tzinfo=timezone.utc)
    minute_ts = int(minute.timestamp())
    start = datetime(2024, 7, 18, 12, 30, 30, tzinfo=timezone.utc)
    end = datetime(2024, 7, 18, 12, 31, tzinfo=timezone.utc)
    samples = [
        (minute_ts, Decimal("100")),
        (minute_ts + 10, Decimal("100")),
        (minute_ts + 40, Decimal("90")),
    ]
    bars = aggregate_samples_to_minute_bars(samples, start=start, end=end)
    assert len(bars) == 1
    assert bars[0].end_ts == minute_ts + 60
    assert bars[0].open == "100"
    assert bars[0].high == "100"
    assert bars[0].low == "90"
    assert bars[0].close == "90"


def test_aggregate_fixture_hour() -> None:
    payload = json.loads((FIXTURES / "cf_brti_hour.json").read_text())
    samples = parse_cf_history_samples(payload)
    start_ts = samples[0][0] - (samples[0][0] % 60)
    start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end = datetime.fromtimestamp(start_ts + 3600, tz=timezone.utc)
    bars = aggregate_samples_to_minute_bars(samples, start=start, end=end)
    assert bars
    first = bars[0]
    assert first.end_ts == start_ts + 60
    assert first.open == "1.11"
    assert first.high == "9.99"
    assert first.low == "0.01"
    assert first.close == "0.01"


def test_fetch_brti_minute_bars_signed(
    kalshi_credentials,
) -> None:
    seen_auth: list[dict[str, str]] = []
    seen_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "cfbenchmarks/history/values" in request.url.path:
            seen_auth.append(
                {
                    "key": request.headers["KALSHI-ACCESS-KEY"],
                    "sig": request.headers["KALSHI-ACCESS-SIGNATURE"],
                    "ts": request.headers["KALSHI-ACCESS-TIMESTAMP"],
                }
            )
            seen_params.append(dict(request.url.params))
            return httpx.Response(
                200,
                json=json.loads((FIXTURES / "cf_brti_hour.json").read_text()),
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    start = datetime(2099, 4, 1, 12, 0, tzinfo=timezone.utc)
    end = datetime(2099, 4, 1, 13, 0, tzinfo=timezone.utc)
    with httpx.Client(base_url=BASE_URL, transport=transport) as http:
        client = KalshiClient(
            http, credentials=kalshi_credentials, max_rps=0
        )
        bars = client.fetch_brti_minute_bars(start=start, end=end)
    assert seen_auth
    assert seen_auth[0]["key"] == "test-api-key-id"
    assert seen_params[0]["timestamp"] == "2099-04-01T12:00:00.000Z"
    assert seen_params[0]["timespan"] == "HOUR"
    assert bars


@pytest.mark.parametrize(
    ("status", "exc", "match"),
    [
        (401, KalshiAuthError, "401"),
        (403, KalshiAuthError, "403"),
        (503, KalshiAuthError, "503"),
        (400, KalshiAPIError, "400"),
    ],
)
def test_fetch_brti_maps_http_errors(
    kalshi_credentials, status: int, exc: type[Exception], match: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"error": {"message": "nope", "details": "bad timestamp"}},
        )

    transport = httpx.MockTransport(handler)
    start = datetime(2024, 7, 18, 12, 0, tzinfo=timezone.utc)
    end = datetime(2024, 7, 18, 13, 0, tzinfo=timezone.utc)
    with httpx.Client(base_url=BASE_URL, transport=transport) as http:
        client = KalshiClient(
            http, credentials=kalshi_credentials, max_rps=0
        )
        with pytest.raises(exc, match=match):
            client.fetch_brti_minute_bars(start=start, end=end)


def test_fetch_requires_credentials() -> None:
    with httpx.Client(base_url=BASE_URL) as http:
        client = KalshiClient(http, max_rps=0)
        with pytest.raises(KalshiCredentialsError):
            client.fetch_brti_minute_bars(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
            )


def test_malformed_private_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.key"
    bad.write_text("not-a-pem\n", encoding="utf-8")
    with pytest.raises(KalshiCredentialsError, match="invalid private key"):
        require_kalshi_credentials(
            env={
                "KALSHI_API_KEY_ID": "x",
                "KALSHI_PRIVATE_KEY_PATH": str(bad),
            }
        )


def test_settled_markets_remain_unsigned(kalshi_credentials) -> None:
    auth_on_settled = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal auth_on_settled
        path = request.url.path
        if "cfbenchmarks" in path:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "serverTime": "2099-01-01T00:00:00.000Z",
                        "payload": [],
                    }
                },
            )
        if any(
            header in request.headers
            for header in (
                "kalshi-access-key",
                "KALSHI-ACCESS-KEY",
            )
        ):
            auth_on_settled = True
        if path.endswith("/historical/cutoff"):
            return httpx.Response(
                200,
                json={
                    "market_settled_ts": "2000-01-01T00:00:00Z",
                    "trades_created_ts": "2000-01-01T00:00:00Z",
                    "orders_updated_ts": "2000-01-01T00:00:00Z",
                },
            )
        if path.endswith("/markets"):
            return httpx.Response(200, json={"cursor": "", "markets": []})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    start = datetime(2099, 4, 1, tzinfo=timezone.utc)
    end = datetime(2099, 4, 1, 1, tzinfo=timezone.utc)
    with httpx.Client(base_url=BASE_URL, transport=transport) as http:
        client = KalshiClient(
            http, credentials=kalshi_credentials, max_rps=0
        )
        client.fetch_settled_kxbtc(start=start, end=end)
    assert auth_on_settled is False
