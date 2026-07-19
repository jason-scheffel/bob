# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from typer.testing import CliRunner

from bob.browse import (
    EventRow,
    format_bracket_range,
    format_btc,
    format_close_et,
    format_event_label,
    load_brackets,
    load_events,
    winning_bracket,
)
from bob.cli import app
from bob.db import (
    MARKET_HOUR_MINUTES,
    MinuteBar,
    acknowledge_candle_hour_gap,
    connect,
    expected_market_minute_ends,
    initialize_schema,
    normalize_market_quote_hour,
    store_btc_candles,
    store_market_candles,
    store_settled_events,
)
from bob.kalshi import (
    STATUS_COMPLETE,
    STATUS_MISSING_EXPIRATION,
    Bracket,
    Event,
    SettledEvent,
    no_markets_event,
)
from bob.viz import (
    HOURS_PER_DAY,
    filter_coverage,
    format_coverage_percent,
    load_coverage,
    months_from_report,
    summarize_report,
)

runner = CliRunner()


def _event(close: datetime, *, ticker: str | None = None) -> SettledEvent:
    day = close.astimezone(timezone.utc)
    event_ticker = ticker or (
        f"KXBTC-{day.year % 100:02d}APR{day.day:02d}{day.hour:02d}"
    )
    return SettledEvent(
        event=Event(
            event_ticker=event_ticker,
            close_ts=close,
            status=STATUS_COMPLETE,
            expiration_value=Decimal("420.69"),
        ),
        brackets=(
            Bracket(
                ticker=f"{event_ticker}-B420",
                event_ticker=event_ticker,
                floor_strike=Decimal("400"),
                cap_strike=Decimal("499.99"),
                won=True,
            ),
        ),
    )


def test_load_coverage_empty() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    report = load_coverage(connection)
    assert report.days == ()
    assert report.total_events == 0
    connection.close()


def test_load_coverage_full_partial_and_gap() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    day1 = [
        _event(
            datetime(2099, 4, 1, hour, tzinfo=timezone.utc),
            ticker=f"KXBTC-99APR01{hour:02d}",
        )
        for hour in range(HOURS_PER_DAY)
    ]
    day3 = [
        _event(
            datetime(2099, 4, 3, hour, tzinfo=timezone.utc),
            ticker=f"KXBTC-99APR03{hour:02d}",
        )
        for hour in range(12)
    ]
    store_settled_events(connection, day1 + day3)
    report = load_coverage(connection)
    assert [day.day.isoformat() for day in report.days] == [
        "2099-04-01",
        "2099-04-02",
        "2099-04-03",
    ]
    # Events full but no candles → partial (green requires both).
    assert report.days[0].status == "partial"
    assert report.days[0].candle_hours == 0
    assert report.days[1].status == "empty"
    assert report.days[2].status == "partial"
    connection.close()


def test_filter_coverage_range() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_settled_events(
        connection,
        [
            _event(
                datetime(2099, 4, day, 0, tzinfo=timezone.utc),
                ticker=f"KXBTC-99APR{day:02d}00",
            )
            for day in (1, 2, 3)
        ],
    )
    report = load_coverage(connection)
    sliced = filter_coverage(report, date(2099, 4, 2), date(2099, 4, 3))
    assert [day.day.isoformat() for day in sliced.days] == [
        "2099-04-02",
        "2099-04-03",
    ]
    assert "2099-04-02 → 2099-04-03" in summarize_report(sliced)
    connection.close()


def test_format_coverage_percent_never_rounds_partial_to_100() -> None:
    # April-like: events full, one candle hour short → mean rounds to 100%
    # with :.0% but status is partial.
    fraction = (1.0 + 719 / 720) / 2
    assert f"{fraction:.0%}" == "100%"
    assert format_coverage_percent(fraction, complete=False) == "99.9%"
    assert format_coverage_percent(1.0, complete=True) == "100%"
    assert format_coverage_percent(0.0, complete=False) == "0%"
    assert format_coverage_percent(0.5, complete=False) == "50%"


def test_months_from_report_groups_and_status() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    events = [
        _event(
            datetime(2099, 3, 31, hour, tzinfo=timezone.utc),
            ticker=f"KXBTC-99MAR31{hour:02d}",
        )
        for hour in range(HOURS_PER_DAY)
    ] + [
        _event(
            datetime(2099, 4, 1, 0, tzinfo=timezone.utc),
            ticker="KXBTC-99APR0100",
        )
    ]
    store_settled_events(connection, events)
    months = months_from_report(load_coverage(connection))
    assert [month.label for month in months] == ["Mar 2099", "Apr 2099"]
    assert months[0].status == "partial"
    assert months[0].covered_events == HOURS_PER_DAY
    assert months[0].covered_candle_hours == 0
    assert months[1].status == "partial"
    assert months[1].days_with_data == 1
    connection.close()


def test_overall_fraction_caps_per_day() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    events = [
        _event(
            datetime(2099, 4, 1, hour % 24, tzinfo=timezone.utc),
            ticker=f"KXBTC-99APR01A{hour:02d}",
        )
        for hour in range(25)
    ] + [
        _event(
            datetime(2099, 4, 2, hour, tzinfo=timezone.utc),
            ticker=f"KXBTC-99APR02{hour:02d}",
        )
        for hour in range(23)
    ]
    store_settled_events(connection, events)
    report = load_coverage(connection)
    assert report.total_events == 48
    assert report.covered_events == 47
    # No candles → combined fraction is half the event fraction.
    assert report.overall_fraction == (47 / 48) / 2
    connection.close()


def test_load_coverage_counts_any_status() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_settled_events(
        connection,
        [
            _event(datetime(2099, 4, 1, 0, tzinfo=timezone.utc)),
            no_markets_event(
                "KXBTC-99APR0101",
                datetime(2099, 4, 1, 1, tzinfo=timezone.utc),
            ),
            SettledEvent(
                event=Event(
                    event_ticker="KXBTC-99APR0102",
                    close_ts=datetime(2099, 4, 1, 2, tzinfo=timezone.utc),
                    status=STATUS_MISSING_EXPIRATION,
                    expiration_value=None,
                ),
                brackets=(
                    Bracket(
                        ticker="KXBTC-99APR0102-B420",
                        event_ticker="KXBTC-99APR0102",
                        floor_strike=Decimal("400"),
                        cap_strike=Decimal("499.99"),
                        won=True,
                    ),
                ),
            ),
        ],
    )
    report = load_coverage(connection)
    assert report.total_events == 3
    assert report.days[0].events == 3
    assert report.days[0].complete == 1
    assert report.days[0].flagged == 2
    assert report.days[0].status == "partial"
    assert report.days[0].label() == "1✓ 2· / 0c / 0q/2"
    assert report.complete_events == 1
    assert report.flagged_events == 2
    assert "1 complete" in summarize_report(report)
    assert "2 flagged" in summarize_report(report)
    connection.close()


def _seed_candle_hours(connection, day: date, hours: range) -> None:
    bars: list[MinuteBar] = []
    for hour in hours:
        hour_start = int(
            datetime(
                day.year, day.month, day.day, hour, tzinfo=timezone.utc
            ).timestamp()
        )
        for offset in range(1, 61):
            bars.append(
                MinuteBar(
                    end_ts=hour_start + offset * 60,
                    open="1",
                    high="1",
                    low="1",
                    close="1",
                )
            )
    store_btc_candles(connection, bars)


def _seed_quote_hours(connection, closes: list[datetime]) -> None:
    """Null-pad complete market quote hours for each event's brackets."""
    for close in closes:
        close_ts = int(close.timestamp())
        tickers = [
            row[0]
            for row in connection.execute(
                """
                SELECT b.ticker FROM brackets b
                JOIN events e ON e.event_ticker = b.event_ticker
                WHERE e.close_ts = ?
                """,
                (close_ts,),
            )
        ]
        for ticker in tickers:
            store_market_candles(
                connection,
                normalize_market_quote_hour(ticker, close_ts, ()),
            )


def test_acknowledged_gap_counts_toward_green() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    day = date(2099, 4, 1)
    events = [
        _event(
            datetime(2099, 4, 1, hour, tzinfo=timezone.utc),
            ticker=f"KXBTC-99APR01{hour:02d}",
        )
        for hour in range(HOURS_PER_DAY)
    ]
    store_settled_events(connection, events)
    _seed_candle_hours(connection, day, range(23))
    # Hour 23 incomplete (10 bars).
    hour = datetime(2099, 4, 1, 23, tzinfo=timezone.utc)
    store_btc_candles(
        connection,
        [
            MinuteBar(
                end_ts=int(hour.timestamp()) + 60 * i,
                open="1",
                high="1",
                low="1",
                close="1",
            )
            for i in range(1, 11)
        ],
    )
    report = load_coverage(connection)
    assert report.days[0].status == "partial"
    assert report.days[0].candle_hours == 23
    assert report.days[0].gappy_hours == 0

    acknowledge_candle_hour_gap(
        connection,
        hour,
        now=datetime(2099, 4, 2, 1, tzinfo=timezone.utc),
    )
    report = load_coverage(connection)
    assert report.days[0].candle_hours == 23
    assert report.days[0].gappy_hours == 1
    assert report.days[0].accounted_candle_hours == 24
    assert report.days[0].status == "partial"
    assert report.days[0].label() == "24✓ / 23c 1g / 0q/24"

    _seed_quote_hours(
        connection,
        [datetime(2099, 4, 1, h, tzinfo=timezone.utc) for h in range(HOURS_PER_DAY)],
    )
    report = load_coverage(connection)
    assert report.days[0].quote_hours == 24
    assert report.days[0].status == "full"
    assert report.days[0].label() == "24✓ / 23c 1g / 24q"
    month = months_from_report(report)[0]
    assert month.status == "full"
    assert month.candles_cell() == "23c+1g/24"
    assert format_coverage_percent(month.overall_fraction, complete=True) == "100%"
    connection.close()


def test_gap_ignored_when_hour_later_complete() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    day = date(2099, 4, 1)
    hour = datetime(2099, 4, 1, 0, tzinfo=timezone.utc)
    store_btc_candles(
        connection,
        [
            MinuteBar(
                end_ts=int(hour.timestamp()) + 60 * i,
                open="1",
                high="1",
                low="1",
                close="1",
            )
            for i in range(1, 11)
        ],
    )
    acknowledge_candle_hour_gap(
        connection,
        hour,
        now=datetime(2099, 4, 2, 1, tzinfo=timezone.utc),
    )
    _seed_candle_hours(connection, day, range(1))
    report = load_coverage(connection)
    assert report.days[0].candle_hours == 1
    assert report.days[0].gappy_hours == 0
    connection.close()


def test_full_day_with_flags_needs_candles_for_green() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    events = [
        _event(
            datetime(2099, 4, 1, hour, tzinfo=timezone.utc),
            ticker=f"KXBTC-99APR01{hour:02d}",
        )
        for hour in range(20)
    ] + [
        no_markets_event(
            f"KXBTC-99APR01{hour:02d}",
            datetime(2099, 4, 1, hour, tzinfo=timezone.utc),
        )
        for hour in range(20, 24)
    ]
    store_settled_events(connection, events)
    report = load_coverage(connection)
    assert report.days[0].status == "partial"
    assert report.days[0].complete == 20
    assert report.days[0].flagged == 4
    assert report.days[0].label() == "20✓ 4· / 0c / 0q/20"
    assert report.unknown_events == 0

    _seed_candle_hours(connection, date(2099, 4, 1), range(24))
    report = load_coverage(connection)
    assert report.days[0].candle_hours == 24
    assert report.days[0].status == "partial"
    assert report.days[0].label() == "20✓ 4· / 24c / 0q/20"

    _seed_quote_hours(
        connection,
        [datetime(2099, 4, 1, h, tzinfo=timezone.utc) for h in range(20)],
    )
    report = load_coverage(connection)
    assert report.days[0].quote_hours == 20
    assert report.days[0].status == "full"
    assert report.days[0].label() == "20✓ 4· / 24c / 20q"
    connection.close()


def test_load_coverage_candle_only_day() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    _seed_candle_hours(connection, date(2099, 5, 1), range(12))
    report = load_coverage(connection)
    assert [day.day.isoformat() for day in report.days] == ["2099-05-01"]
    assert report.days[0].events == 0
    assert report.days[0].candle_hours == 12
    assert report.days[0].status == "partial"
    assert report.days[0].label() == "0✓ / 12c / 0q"
    connection.close()


def test_load_coverage_counts_quote_complete_hours() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    close = datetime(2099, 4, 1, 12, tzinfo=timezone.utc)
    store_settled_events(connection, [_event(close)])
    close_ts = int(close.timestamp())
    ticker = "KXBTC-99APR0112-B420"
    hour = normalize_market_quote_hour(ticker, close_ts, ())
    assert len(hour) == MARKET_HOUR_MINUTES
    assert len(expected_market_minute_ends(close_ts)) == MARKET_HOUR_MINUTES
    store_market_candles(connection, hour)
    report = load_coverage(connection)
    assert report.days[0].quote_expected == 1
    assert report.days[0].quote_hours == 1
    assert report.days[0].label() == "1✓ / 0c / 1q"
    assert "quotes 1/1" in summarize_report(report)
    connection.close()


def test_load_events_skips_non_complete() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_settled_events(
        connection,
        [
            _event(datetime(2099, 4, 1, 12, tzinfo=timezone.utc)),
            no_markets_event(
                "KXBTC-99APR0113",
                datetime(2099, 4, 1, 13, tzinfo=timezone.utc),
            ),
        ],
    )
    events = load_events(
        connection,
        datetime(2099, 4, 1, tzinfo=timezone.utc),
        datetime(2099, 4, 2, tzinfo=timezone.utc),
    )
    assert [event.event_ticker for event in events] == ["KXBTC-99APR0112"]
    connection.close()


def test_load_events_half_open_range() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_settled_events(
        connection,
        [
            _event(
                datetime(2099, 4, day, 12, tzinfo=timezone.utc),
                ticker=f"KXBTC-99APR{day:02d}12",
            )
            for day in (1, 2, 3)
        ],
    )
    events = load_events(
        connection,
        datetime(2099, 4, 1, tzinfo=timezone.utc),
        datetime(2099, 4, 3, tzinfo=timezone.utc),
    )
    assert [event.event_ticker for event in events] == [
        "KXBTC-99APR0112",
        "KXBTC-99APR0212",
    ]
    connection.close()


def test_load_brackets_for_event() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_settled_events(
        connection,
        [_event(datetime(2099, 4, 1, 0, tzinfo=timezone.utc))],
    )
    brackets = load_brackets(connection, "KXBTC-99APR0100")
    assert len(brackets) == 1
    assert brackets[0].ticker == "KXBTC-99APR0100-B420"
    assert brackets[0].won is True
    assert brackets[0].floor_strike == "400"
    assert winning_bracket(brackets) is brackets[0]
    assert load_brackets(connection, "missing") == ()
    connection.close()


def test_browse_formatters_are_human_readable() -> None:
    close = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    assert format_close_et(close) == "Jun 30, 2026 8:00 PM ET"
    assert format_btc("56562.29") == "$56,562.29"
    assert format_bracket_range("50200", "50299.99") == "$50,200 – $50,299.99"
    assert format_bracket_range(None, "50000") == "below $50,000"
    assert format_bracket_range("60000", None) == "$60,000 and above"
    event = EventRow(
        event_ticker="KXBTC-26JUN3020",
        close_ts=close,
        expiration_value="56562.29",
    )
    assert format_event_label(event) == ("Jun 30, 2026 8:00 PM ET · BTC $56,562.29")


def test_cli_viz_missing_db(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    missing = tmp_path / "missing.sqlite"
    result = runner.invoke(app, ["viz", "--db", str(missing)])
    assert result.exit_code == 2
    assert "database not found" in result.output


def test_cli_viz_rejects_memory_db(monkeypatch) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    result = runner.invoke(app, ["viz", "--db", ":memory:"])
    assert result.exit_code == 2
    assert "database not found" in result.output


def test_cli_viz_requires_gate_and_launches_streamlit(
    monkeypatch, tmp_path: Path
) -> None:
    called = {"gate": False, "streamlit": False}

    def fake_gate() -> None:
        called["gate"] = True

    def fake_streamlit(db: Path) -> int:
        called["streamlit"] = True
        assert db.name == "viz.sqlite"
        assert db.is_file()
        return 0

    monkeypatch.setattr("bob.cli.require_gate", fake_gate)
    monkeypatch.setattr("bob.cli.run_streamlit", fake_streamlit)
    db_path = tmp_path / "viz.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(
        connection,
        [_event(datetime(2099, 4, 1, 0, tzinfo=timezone.utc))],
    )
    connection.close()
    result = runner.invoke(app, ["viz", "--db", str(db_path)])
    assert result.exit_code == 0
    assert called["gate"] is True
    assert called["streamlit"] is True
