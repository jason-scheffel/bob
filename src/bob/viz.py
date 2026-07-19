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

    @property
    def events(self) -> int:
        """Accounted hours (any status)."""
        return self.complete + self.flagged

    @property
    def expected(self) -> int:
        return HOURS_PER_DAY

    @property
    def unknown(self) -> int:
        return max(0, self.expected - self.events)

    @property
    def status(self) -> str:
        """Color status from accounted hours, not usable BTC prints."""
        if self.events <= 0:
            return "empty"
        if self.events >= self.expected:
            return "full"
        return "partial"

    def label(self) -> str:
        """Short cell label: complete✓ and flagged· when present."""
        body = f"{self.complete}✓"
        if self.flagged:
            body += f" {self.flagged}·"
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
        return sum(1 for day in self.days if day.events > 0)

    @property
    def missing_days(self) -> int:
        return sum(1 for day in self.days if day.events <= 0)

    @property
    def expected_events(self) -> int:
        return len(self.days) * HOURS_PER_DAY

    @property
    def covered_events(self) -> int:
        """Accounted hours toward coverage, capped at 24 per day."""
        return sum(min(day.events, day.expected) for day in self.days)

    @property
    def complete_events(self) -> int:
        return sum(day.complete for day in self.days)

    @property
    def flagged_events(self) -> int:
        return sum(day.flagged for day in self.days)

    @property
    def unknown_events(self) -> int:
        return max(0, self.expected_events - self.covered_events)

    @property
    def overall_fraction(self) -> float:
        if self.expected_events == 0:
            return 0.0
        return self.covered_events / self.expected_events


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
        return sum(1 for day in self.days if day.events > 0)

    @property
    def missing_days(self) -> int:
        return sum(1 for day in self.days if day.events <= 0)

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
    def overall_fraction(self) -> float:
        if self.expected_events == 0:
            return 0.0
        return self.covered_events / self.expected_events

    @property
    def status(self) -> str:
        if not self.days or self.covered_events <= 0:
            return "empty"
        if self.covered_events >= self.expected_events:
            return "full"
        return "partial"


def load_coverage(connection: sqlite3.Connection) -> CoverageReport:
    rows = connection.execute(
        "SELECT close_ts, status FROM events ORDER BY close_ts"
    ).fetchall()
    if not rows:
        return CoverageReport(days=(), total_events=0)

    complete: dict[date, int] = defaultdict(int)
    flagged: dict[date, int] = defaultdict(int)
    for close_ts, status in rows:
        day = datetime.fromtimestamp(int(close_ts), tz=timezone.utc).date()
        if status == STATUS_COMPLETE:
            complete[day] += 1
        else:
            flagged[day] += 1

    first = min(complete.keys() | flagged.keys())
    last = max(complete.keys() | flagged.keys())
    days: list[DayCoverage] = []
    cursor = first
    while cursor <= last:
        days.append(
            DayCoverage(
                day=cursor,
                complete=complete.get(cursor, 0),
                flagged=flagged.get(cursor, 0),
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
        return "No events in selection."
    return (
        f"{label}: {report.first_day} → {report.last_day}  ·  "
        f"{report.complete_events} complete  ·  "
        f"{report.flagged_events} flagged  ·  "
        f"{report.unknown_events} unknown  ·  "
        f"accounted {report.covered_events}/{report.expected_events} "
        f"({report.overall_fraction:.0%})"
    )
