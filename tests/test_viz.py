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
from bob.db import connect, initialize_schema, store_settled_events
from bob.kalshi import Bracket, Event, SettledEvent
from bob.viz import (
    HOURS_PER_DAY,
    filter_coverage,
    load_coverage,
    months_from_report,
    summarize_report,
)

runner = CliRunner()


def _event(close: datetime, *, ticker: str | None = None) -> SettledEvent:
    day = close.astimezone(timezone.utc)
    event_ticker = ticker or (
        f"KXBTC-{day.year % 100:02d}APR"
        f"{day.day:02d}{day.hour:02d}"
    )
    return SettledEvent(
        event=Event(
            event_ticker=event_ticker,
            close_ts=close,
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
    assert report.days[0].status == "full"
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
    assert months[0].status == "full"
    assert months[0].covered_events == HOURS_PER_DAY
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
    assert report.overall_fraction == 47 / 48
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
    assert format_event_label(event) == (
        "Jun 30, 2026 8:00 PM ET · BTC $56,562.29"
    )


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
