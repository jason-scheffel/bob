# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone

from bob.db import MinuteBar, store_btc_candles, utc_hour_starts


def seed_complete_candle_hours(
    connection,
    start: datetime,
    end: datetime,
) -> int:
    """Write full 60-minute coverage for every UTC hour overlapping ``[start, end)``."""
    bars: list[MinuteBar] = []
    for hour_start in utc_hour_starts(start, end):
        start_ts = int(hour_start.astimezone(timezone.utc).timestamp())
        for offset in range(1, 61):
            bars.append(
                MinuteBar(
                    end_ts=start_ts + offset * 60,
                    open="1",
                    high="1",
                    low="1",
                    close="1",
                )
            )
    return store_btc_candles(connection, bars)


def cf_empty_response() -> dict:
    return {
        "_fixture": "SYNTHETIC — empty CF history (not real)",
        "data": {
            "serverTime": "2099-01-01T00:00:00.000Z",
            "payload": [],
        },
    }
