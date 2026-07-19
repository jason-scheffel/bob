# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from bob.kalshi import KalshiCredentials, load_private_key


@pytest.fixture
def rsa_pem(tmp_path: Path) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path = tmp_path / "kalshi-test.key"
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return path


@pytest.fixture
def kalshi_credentials(
    rsa_pem: Path, monkeypatch: pytest.MonkeyPatch
) -> KalshiCredentials:
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-api-key-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(rsa_pem))
    return KalshiCredentials(
        api_key_id="test-api-key-id",
        private_key=load_private_key(rsa_pem),
        base_url="https://external-api.kalshi.com/trade-api/v2",
    )
