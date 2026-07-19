# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from bob.kalshi import STATUS_COMPLETE

HOURS_PER_DAY = 24


def format_coverage_percent(fraction: float, *, complete: bool) -> str:
    """Percent label aligned with full/partial status.

    Incomplete coverage never displays as 100% (``:.0%`` would round
    719/720-style means up).
    """
    if complete:
        return "100%"
    if fraction <= 0:
        return "0%"
    tenths = int(fraction * 1000) / 10
    if tenths >= 100:
        tenths = 99.9
    if tenths == int(tenths):
        return f"{int(tenths)}%"
    return f"{tenths:.1f}%"


@dataclass(frozen=True, slots=True)
class DayCoverage:
    day: date
    complete: int
    flagged: int
    candle_hours: int = 0
    gappy_hours: int = 0

    @property
    def events(self) -> int:
        """Accounted event hours (any status)."""
        return self.complete + self.flagged

    @property
    def accounted_candle_hours(self) -> int:
        """Complete 60/60 hours plus acknowledged upstream gaps."""
        return self.candle_hours + self.gappy_hours

    @property
    def expected(self) -> int:
        return HOURS_PER_DAY

    @property
    def unknown(self) -> int:
        return max(0, self.expected - self.events)

    @property
    def status(self) -> str:
        """Green when events and accounted candles both meet expected."""
        if self.events <= 0 and self.accounted_candle_hours <= 0:
            return "empty"
        events_full = self.events >= self.expected
        candles_full = self.accounted_candle_hours >= self.expected
        if events_full and candles_full:
            return "full"
        return "partial"

    def label(self) -> str:
        """Short cell label: events then candles (c) and gaps (g)."""
        body = f"{self.complete}✓"
        if self.flagged:
            body += f" {self.flagged}·"
        body += f" / {self.candle_hours}c"
        if self.gappy_hours:
            body += f" {self.gappy_hours}g"
        return body

    def candles_cell(self, *, expected: int | None = None) -> str:
        denom = self.expected if expected is None else expected
        if self.gappy_hours:
            return f"{self.candle_hours}c+{self.gappy_hours}g/{denom}"
        return f"{self.accounted_candle_hours}/{denom}"


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
            1 for day in self.days if day.events > 0 or day.accounted_candle_hours > 0
        )

    @property
    def missing_days(self) -> int:
        return sum(
            1
            for day in self.days
            if day.events <= 0 and day.accounted_candle_hours <= 0
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
    def gappy_hours(self) -> int:
        return sum(day.gappy_hours for day in self.days)

    @property
    def covered_candle_hours(self) -> int:
        """Accounted candle hours (complete + gappy), capped at 24/day."""
        return sum(min(day.accounted_candle_hours, day.expected) for day in self.days)

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
            1 for day in self.days if day.events > 0 or day.accounted_candle_hours > 0
        )

    @property
    def missing_days(self) -> int:
        return sum(
            1
            for day in self.days
            if day.events <= 0 and day.accounted_candle_hours <= 0
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
    def gappy_hours(self) -> int:
        return sum(day.gappy_hours for day in self.days)

    @property
    def covered_candle_hours(self) -> int:
        return sum(min(day.accounted_candle_hours, day.expected) for day in self.days)

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

    def candles_cell(self) -> str:
        if self.gappy_hours:
            return f"{self.candle_hours}c+{self.gappy_hours}g/{self.expected_events}"
        return f"{self.covered_candle_hours}/{self.expected_events}"


def _complete_hour_starts(
    connection: sqlite3.Connection,
) -> set[int]:
    rows = connection.execute(
        """
        SELECT ((end_ts - 1) / 3600) * 3600 AS hour_start
        FROM btc_candles
        GROUP BY hour_start
        HAVING COUNT(*) = 60
        """
    ).fetchall()
    return {int(hour_start) for (hour_start,) in rows}


def _complete_candle_hours_by_day(
    complete_hours: set[int],
) -> dict[date, int]:
    by_day: dict[date, int] = defaultdict(int)
    for hour_start in complete_hours:
        day = datetime.fromtimestamp(hour_start, tz=timezone.utc).date()
        by_day[day] += 1
    return by_day


def _gappy_candle_hours_by_day(
    connection: sqlite3.Connection,
    complete_hours: set[int],
) -> dict[date, int]:
    """Acknowledged gaps, excluding hours that later became complete."""
    rows = connection.execute("SELECT hour_start FROM candle_hour_gaps").fetchall()
    by_day: dict[date, int] = defaultdict(int)
    for (hour_start,) in rows:
        ts = int(hour_start)
        if ts in complete_hours:
            continue
        day = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        by_day[day] += 1
    return by_day


def load_coverage(connection: sqlite3.Connection) -> CoverageReport:
    event_rows = connection.execute(
        "SELECT close_ts, status FROM events ORDER BY close_ts"
    ).fetchall()
    complete_hours = _complete_hour_starts(connection)
    candle_by_day = _complete_candle_hours_by_day(complete_hours)
    gappy_by_day = _gappy_candle_hours_by_day(connection, complete_hours)

    complete: dict[date, int] = defaultdict(int)
    flagged: dict[date, int] = defaultdict(int)
    for close_ts, status in event_rows:
        day = datetime.fromtimestamp(int(close_ts), tz=timezone.utc).date()
        if status == STATUS_COMPLETE:
            complete[day] += 1
        else:
            flagged[day] += 1

    days_present = set(complete) | set(flagged) | set(candle_by_day) | set(gappy_by_day)
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
                gappy_hours=gappy_by_day.get(cursor, 0),
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
    complete = (
        report.covered_events >= report.expected_events
        and report.covered_candle_hours >= report.expected_events
    )
    combined = format_coverage_percent(report.overall_fraction, complete=complete)
    if report.gappy_hours:
        candles = (
            f"{report.candle_hours}c+{report.gappy_hours}g/{report.expected_events}"
        )
    else:
        candles = f"{report.covered_candle_hours}/{report.expected_events}"
    return (
        f"{label}: {report.first_day} → {report.last_day}  ·  "
        f"{report.complete_events} complete  ·  "
        f"{report.flagged_events} flagged  ·  "
        f"{report.unknown_events} unknown  ·  "
        f"candles {candles}  ·  "
        f"accounted {report.covered_events}/{report.expected_events} events  ·  "
        f"combined {combined}"
    )
