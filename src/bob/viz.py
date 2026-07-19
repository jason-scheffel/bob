# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from bob.kalshi import STATUS_COMPLETE

HOURS_PER_DAY = 24


@dataclass(frozen=True, slots=True)
class DayCoverage:
    day: date
    complete: int
    flagged: int
    candle_hours: int = 0

    @property
    def events(self) -> int:
        """Accounted event hours (any status)."""
        return self.complete + self.flagged

    @property
    def expected(self) -> int:
        return HOURS_PER_DAY

    @property
    def unknown(self) -> int:
        return max(0, self.expected - self.events)

    @property
    def status(self) -> str:
        """Green only when both events and candles are fully accounted."""
        if self.events <= 0 and self.candle_hours <= 0:
            return "empty"
        events_full = self.events >= self.expected
        candles_full = self.candle_hours >= self.expected
        if events_full and candles_full:
            return "full"
        return "partial"

    def label(self) -> str:
        """Short cell label: events then complete candle hours."""
        body = f"{self.complete}✓"
        if self.flagged:
            body += f" {self.flagged}·"
        body += f" / {self.candle_hours}c"
        return body


@dataclass(frozen=True, slots=True)
class CoverageReport:
    days: tuple[DayCoverage, ...]
    total_events: int

    @property
    def first_day(self) -> date | None:
        return self.days[0].day if self.days else None

    @property
    def last_day(self) -> date | None:
        return self.days[-1].day if self.days else None

    @property
    def days_with_data(self) -> int:
        return sum(
            1
            for day in self.days
            if day.events > 0 or day.candle_hours > 0
        )

    @property
    def missing_days(self) -> int:
        return sum(
            1
            for day in self.days
            if day.events <= 0 and day.candle_hours <= 0
        )

    @property
    def expected_events(self) -> int:
        return len(self.days) * HOURS_PER_DAY

    @property
    def covered_events(self) -> int:
        """Accounted event hours toward coverage, capped at 24 per day."""
        return sum(min(day.events, day.expected) for day in self.days)

    @property
    def complete_events(self) -> int:
        return sum(day.complete for day in self.days)

    @property
    def flagged_events(self) -> int:
        return sum(day.flagged for day in self.days)

    @property
    def candle_hours(self) -> int:
        return sum(day.candle_hours for day in self.days)

    @property
    def covered_candle_hours(self) -> int:
        return sum(min(day.candle_hours, day.expected) for day in self.days)

    @property
    def unknown_events(self) -> int:
        return max(0, self.expected_events - self.covered_events)

    @property
    def unknown_candle_hours(self) -> int:
        return max(0, self.expected_events - self.covered_candle_hours)

    @property
    def overall_fraction(self) -> float:
        """Mean of event and candle accounted fractions."""
        if self.expected_events == 0:
            return 0.0
        event_frac = self.covered_events / self.expected_events
        candle_frac = self.covered_candle_hours / self.expected_events
        return (event_frac + candle_frac) / 2.0


@dataclass(frozen=True, slots=True)
class MonthCoverage:
    year: int
    month: int
    days: tuple[DayCoverage, ...]

    @property
    def label(self) -> str:
        return date(self.year, self.month, 1).strftime("%b %Y")

    @property
    def days_with_data(self) -> int:
        return sum(
            1
            for day in self.days
            if day.events > 0 or day.candle_hours > 0
        )

    @property
    def missing_days(self) -> int:
        return sum(
            1
            for day in self.days
            if day.events <= 0 and day.candle_hours <= 0
        )

    @property
    def expected_events(self) -> int:
        return len(self.days) * HOURS_PER_DAY

    @property
    def covered_events(self) -> int:
        return sum(min(day.events, day.expected) for day in self.days)

    @property
    def complete_events(self) -> int:
        return sum(day.complete for day in self.days)

    @property
    def flagged_events(self) -> int:
        return sum(day.flagged for day in self.days)

    @property
    def candle_hours(self) -> int:
        return sum(day.candle_hours for day in self.days)

    @property
    def covered_candle_hours(self) -> int:
        return sum(min(day.candle_hours, day.expected) for day in self.days)

    @property
    def overall_fraction(self) -> float:
        if self.expected_events == 0:
            return 0.0
        event_frac = self.covered_events / self.expected_events
        candle_frac = self.covered_candle_hours / self.expected_events
        return (event_frac + candle_frac) / 2.0

    @property
    def status(self) -> str:
        if not self.days or (
            self.covered_events <= 0 and self.covered_candle_hours <= 0
        ):
            return "empty"
        if (
            self.covered_events >= self.expected_events
            and self.covered_candle_hours >= self.expected_events
        ):
            return "full"
        return "partial"


def _complete_candle_hours_by_day(
    connection: sqlite3.Connection,
) -> dict[date, int]:
    """Count UTC hours with all 60 minute bars present."""
    rows = connection.execute(
        """
        SELECT ((end_ts - 1) / 3600) * 3600 AS hour_start, COUNT(*) AS n
        FROM btc_candles
        GROUP BY hour_start
        HAVING n = 60
        """
    ).fetchall()
    by_day: dict[date, int] = defaultdict(int)
    for hour_start, _count in rows:
        day = datetime.fromtimestamp(int(hour_start), tz=timezone.utc).date()
        by_day[day] += 1
    return by_day


def load_coverage(connection: sqlite3.Connection) -> CoverageReport:
    event_rows = connection.execute(
        "SELECT close_ts, status FROM events ORDER BY close_ts"
    ).fetchall()
    candle_by_day = _complete_candle_hours_by_day(connection)

    complete: dict[date, int] = defaultdict(int)
    flagged: dict[date, int] = defaultdict(int)
    for close_ts, status in event_rows:
        day = datetime.fromtimestamp(int(close_ts), tz=timezone.utc).date()
        if status == STATUS_COMPLETE:
            complete[day] += 1
        else:
            flagged[day] += 1

    days_present = set(complete) | set(flagged) | set(candle_by_day)
    if not days_present:
        return CoverageReport(days=(), total_events=0)

    first = min(days_present)
    last = max(days_present)
    days: list[DayCoverage] = []
    cursor = first
    while cursor <= last:
        days.append(
            DayCoverage(
                day=cursor,
                complete=complete.get(cursor, 0),
                flagged=flagged.get(cursor, 0),
                candle_hours=candle_by_day.get(cursor, 0),
            )
        )
        cursor += timedelta(days=1)

    return CoverageReport(
        days=tuple(days),
        total_events=sum(day.events for day in days),
    )


def filter_coverage(
    report: CoverageReport,
    start: date | None,
    end: date | None,
) -> CoverageReport:
    """Return days in ``[start, end]`` (inclusive). ``None`` means unbounded."""
    if not report.days:
        return report
    lo = start if start is not None else report.days[0].day
    hi = end if end is not None else report.days[-1].day
    if lo > hi:
        lo, hi = hi, lo
    days = tuple(day for day in report.days if lo <= day.day <= hi)
    return CoverageReport(
        days=days,
        total_events=sum(day.events for day in days),
    )


def months_from_report(report: CoverageReport) -> tuple[MonthCoverage, ...]:
    """Group coverage days into calendar months (UTC), oldest first."""
    if not report.days:
        return ()
    groups: dict[tuple[int, int], list[DayCoverage]] = {}
    for day in report.days:
        key = (day.day.year, day.day.month)
        groups.setdefault(key, []).append(day)
    return tuple(
        MonthCoverage(year=year, month=month, days=tuple(days))
        for (year, month), days in sorted(groups.items())
    )


def summarize_report(report: CoverageReport, *, label: str = "Range") -> str:
    if not report.days:
        return "No events or candles in selection."
    return (
        f"{label}: {report.first_day} → {report.last_day}  ·  "
        f"{report.complete_events} complete  ·  "
        f"{report.flagged_events} flagged  ·  "
        f"{report.unknown_events} unknown  ·  "
        f"candles {report.covered_candle_hours}/{report.expected_events}  ·  "
        f"accounted {report.covered_events}/{report.expected_events} events  ·  "
        f"combined {report.overall_fraction:.0%}"
    )
