# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import calendar
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from bob.browse import (
    format_btc,
    format_bracket_range,
    format_close_et,
    format_event_label,
    load_brackets,
    load_events,
    winning_bracket,
)
from bob.db import connect
from bob.viz import (
    CoverageReport,
    DayCoverage,
    MonthCoverage,
    filter_coverage,
    format_coverage_percent,
    load_coverage,
    months_from_report,
    summarize_report,
)

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_CAL = calendar.Calendar(firstweekday=calendar.MONDAY)

_STATUS_COLORS = {
    "full": "#2d6a4f",
    "partial": "#b08900",
    "empty": "#9b2226",
    "none": "#6c757d",
}


def _db_path() -> Path:
    raw = os.environ.get("BOB_DB")
    if not raw:
        st.error("BOB_DB is not set. Launch via `bob viz`.")
        st.stop()
    path = Path(raw)
    if not path.is_file():
        st.error(f"Database not found: {path}")
        st.stop()
    return path


def _styled_day_calendar(
    year: int,
    month: int,
    by_day: dict[date, DayCoverage],
):
    weeks = _CAL.monthdatescalendar(year, month)
    rows: list[dict[str, str]] = []
    for week in weeks:
        row: dict[str, str] = {}
        for name, day in zip(_WEEKDAYS, week, strict=True):
            if day.month != month:
                row[name] = ""
                continue
            coverage = by_day.get(day)
            if coverage is None:
                row[name] = f"{day.day}\n—"
            else:
                row[name] = f"{day.day}\n{coverage.label()}"
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(_WEEKDAYS))

    def _apply(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for r, week in enumerate(weeks):
            for c, day in enumerate(week):
                if day.month != month or not data.iat[r, c]:
                    styles.iat[r, c] = "color: #6c757d"
                    continue
                coverage = by_day.get(day)
                status = "none" if coverage is None else coverage.status
                styles.iat[r, c] = (
                    f"background-color: {_STATUS_COLORS[status]}; color: white"
                )
        return styles

    return frame.style.apply(_apply, axis=None)


def _style_months_table(frame: pd.DataFrame, months: tuple[MonthCoverage, ...]):
    def _apply(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for i, month in enumerate(months):
            color = _STATUS_COLORS[month.status]
            styles.iat[i, 0] = f"background-color: {color}; color: white"
        return styles

    return frame.style.apply(_apply, axis=None)


def _coverage_tab(report: CoverageReport) -> None:
    if not report.days:
        st.info("No events or candles in the database.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        start = st.date_input(
            "Filter start",
            value=report.first_day,
            min_value=report.first_day,
            max_value=report.last_day,
            key="cov_start",
        )
    with col_b:
        end = st.date_input(
            "Filter end",
            value=report.last_day,
            min_value=report.first_day,
            max_value=report.last_day,
            key="cov_end",
        )
    selected = filter_coverage(report, start, end)
    st.write(summarize_report(selected))

    level = st.radio(
        "Level",
        ("Months", "Days"),
        horizontal=True,
        key="cov_level",
    )
    if level == "Months":
        _show_months(selected)
    else:
        _show_days(selected)
    st.caption(
        "Cell: events then candles (e.g. 20✓ 4· / 22c 1g). "
        "✓ complete event, · flagged event, "
        "c = hours with all 60 BRTI minutes, "
        "g = acknowledged upstream candle gap. "
        "Green = events and candles both accounted, "
        "yellow = partial, red = empty"
        + (", gray = outside DB span" if level == "Days" else "")
        + "."
    )


def _show_months(report: CoverageReport) -> None:
    months = months_from_report(report)
    if not months:
        st.info("No months in selection.")
        return
    frame = pd.DataFrame(
        [
            {
                "month": month.label,
                "days_with_data": month.days_with_data,
                "empty_days": month.missing_days,
                "complete": month.complete_events,
                "flagged": month.flagged_events,
                "events": (f"{month.covered_events}/{month.expected_events}"),
                "candles": month.candles_cell(),
                "coverage": format_coverage_percent(
                    month.overall_fraction,
                    complete=month.status == "full",
                ),
                "status": month.status,
            }
            for month in months
        ]
    )
    st.dataframe(
        _style_months_table(frame, months),
        width="stretch",
        hide_index=True,
    )


def _show_days(report: CoverageReport) -> None:
    if not report.days:
        st.info("No days in selection.")
        return
    by_day = {day.day: day for day in report.days}
    first = report.first_day
    last = report.last_day
    assert first is not None and last is not None
    default_month = date(last.year, last.month, 1)
    month_value = st.date_input(
        "Month",
        value=default_month,
        min_value=date(first.year, first.month, 1),
        max_value=default_month,
        key="cov_month",
    )
    st.dataframe(
        _styled_day_calendar(month_value.year, month_value.month, by_day),
        width="stretch",
        hide_index=True,
    )


def _browse_tab(connection, report: CoverageReport) -> None:
    if not report.days:
        st.info("No events in the database.")
        return

    last = report.last_day
    first = report.first_day
    assert first is not None and last is not None

    col_a, col_b = st.columns(2)
    with col_a:
        start_day = st.date_input(
            "Start day",
            value=last,
            min_value=first,
            max_value=last,
            key="browse_start",
        )
    with col_b:
        end_day = st.date_input(
            "End day (inclusive)",
            value=last,
            min_value=first,
            max_value=last,
            key="browse_end",
        )
    if start_day > end_day:
        start_day, end_day = end_day, start_day

    start_dt = datetime(
        start_day.year, start_day.month, start_day.day, tzinfo=timezone.utc
    )
    end_dt = datetime(
        end_day.year, end_day.month, end_day.day, tzinfo=timezone.utc
    ) + timedelta(days=1)

    events = load_events(connection, start_dt, end_dt)
    if not events:
        st.warning("No events in this range.")
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Close (ET)": format_close_et(event.close_ts),
                    "Settled BTC": format_btc(event.expiration_value),
                }
                for event in events
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    labels = {format_event_label(event): event.event_ticker for event in events}
    chosen = labels[st.selectbox("Event", list(labels))]
    brackets = load_brackets(connection, chosen)
    if not brackets:
        st.warning("No brackets for this event.")
        return

    winner = winning_bracket(brackets)
    if winner is None:
        st.warning("No single winning bracket recorded for this event.")
    else:
        st.success(
            "Winning range: "
            f"{format_bracket_range(winner.floor_strike, winner.cap_strike)}"
        )

    brackets_frame = pd.DataFrame(
        [
            {
                "Range": format_bracket_range(bracket.floor_strike, bracket.cap_strike),
                "Result": "Won" if bracket.won else "Lost",
            }
            for bracket in brackets
        ]
    )

    def _style_result(column: pd.Series) -> list[str]:
        return [
            "background-color: #2d6a4f; color: white" if value == "Won" else ""
            for value in column
        ]

    st.dataframe(
        brackets_frame.style.apply(_style_result, subset=["Result"]),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Each row is a BTC price bracket for that hour. "
        "Exactly one should be Won (the range that contained the settled price)."
    )


def main() -> None:
    st.set_page_config(page_title="Bob coverage", layout="wide")
    st.title("Bob")
    path = _db_path()
    st.caption(f"DB: {path}")

    connection = connect(path)
    try:
        report = load_coverage(connection)
        coverage, browse = st.tabs(["Coverage", "Browse"])
        with coverage:
            _coverage_tab(report)
        with browse:
            _browse_tab(connection, report)
    finally:
        connection.close()


main()
