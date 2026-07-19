# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from bob.db import (
    StoreCounts,
    connect,
    event_tickers_in_close_range,
    initialize_schema,
    store_settled_events,
)
from bob.gate import require_gate
from bob.kalshi import (
    BASE_URL,
    DEFAULT_MAX_RPS,
    KalshiClient,
    expected_kxbtc_event_tickers,
)

DEFAULT_DB = Path("data/bob.sqlite")

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
)
console = Console(stderr=True)


@app.callback()
def _root() -> None:
    """Bob trading research toolkit."""


def parse_iso_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise typer.BadParameter(f"invalid ISO datetime: {value!r}") from exc
    if parsed.tzinfo is None:
        raise typer.BadParameter(
            f"datetime must include timezone or Z: {value!r}"
        )
    return parsed.astimezone(timezone.utc)


def iter_utc_day_chunks(
    start: datetime, end: datetime
) -> Iterator[tuple[datetime, datetime]]:
    """Yield half-open ``[chunk_start, chunk_end)`` slices on UTC midnights."""
    cursor = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    while cursor < end_utc:
        next_midnight = cursor.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        chunk_end = min(end_utc, next_midnight)
        yield cursor, chunk_end
        cursor = chunk_end


def run_backfill(
    connection,
    client: KalshiClient,
    start: datetime,
    end: datetime,
    *,
    force: bool = False,
    on_day_start=None,
    on_day=None,
    on_skip=None,
) -> StoreCounts:
    total_events = 0
    total_brackets = 0
    chunks = list(iter_utc_day_chunks(start, end))
    known = (
        set()
        if force
        else event_tickers_in_close_range(connection, start, end)
    )
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        if on_day_start is not None:
            on_day_start(
                index=index,
                days=len(chunks),
                day_start=chunk_start,
            )
        expected = expected_kxbtc_event_tickers(chunk_start, chunk_end)
        only_event_tickers: frozenset[str] | None = None
        already = 0
        if not force:
            existing = expected & known
            missing = expected - known
            already = len(existing)
            if not missing:
                if on_day is not None:
                    on_day(
                        index=index,
                        days=len(chunks),
                        day_start=chunk_start,
                        counts=StoreCounts(events=0, brackets=0),
                        requests=0,
                        retries_429=0,
                        already=already,
                        missing=0,
                    )
                continue
            only_event_tickers = frozenset(missing)
        req_before = client.requests
        retries_before = client.retries_429
        events = client.fetch_settled_kxbtc(
            start=chunk_start,
            end=chunk_end,
            on_skip=on_skip,
            only_event_tickers=only_event_tickers,
        )
        counts = store_settled_events(connection, events)
        if not force:
            known.update(item.event.event_ticker for item in events)
        total_events += counts.events
        total_brackets += counts.brackets
        if on_day is not None:
            on_day(
                index=index,
                days=len(chunks),
                day_start=chunk_start,
                counts=counts,
                requests=client.requests - req_before,
                retries_429=client.retries_429 - retries_before,
                already=already,
                missing=(
                    len(only_event_tickers)
                    if only_event_tickers is not None
                    else len(expected)
                ),
            )
    return StoreCounts(events=total_events, brackets=total_brackets)


def _day_label(day_start: datetime) -> str:
    return day_start.astimezone(timezone.utc).strftime("%Y-%m-%d")


def format_eta(seconds: float) -> str:
    """Format a remaining-time estimate for day-line status."""
    if seconds < 0 or seconds != seconds:  # NaN
        return "?"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


@app.command()
def backfill(
    start: Annotated[
        datetime,
        typer.Option(
            ...,
            help="UTC start of close_time range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ],
    end: Annotated[
        datetime,
        typer.Option(
            ...,
            help="UTC end of close_time range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ],
    db: Annotated[
        Path,
        typer.Option(help="SQLite path."),
    ] = DEFAULT_DB,
    rps: Annotated[
        float,
        typer.Option(
            help=(
                "Max HTTP requests/sec (Basic Kalshi Read ≈ 20). "
                "Use 0 to disable pacing."
            ),
        ),
    ] = DEFAULT_MAX_RPS,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Refetch every hour even when it already exists in SQLite.",
        ),
    ] = False,
) -> None:
    """Backfill settled KXBTC events and brackets into SQLite."""
    require_gate()
    if start >= end:
        console.print("[red]Error:[/red] --start must be earlier than --end")
        raise typer.Exit(code=2)
    if rps < 0:
        console.print("[red]Error:[/red] --rps must be >= 0")
        raise typer.Exit(code=2)

    days = len(list(iter_utc_day_chunks(start, end)))
    started = time.monotonic()

    def on_skip(event_ticker: str, reason: str) -> None:
        console.print(f"[yellow]skip[/yellow] {event_ticker}: {reason}")

    connection = connect(db)
    try:
        initialize_schema(connection)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True, elapsed_when_finished=True),
            console=console,
            transient=False,
            expand=True,
        ) as progress:
            task_id = progress.add_task("days", total=days)
            skip_days = 0
            skip_hours = 0

            def flush_skips() -> None:
                nonlocal skip_days, skip_hours
                if skip_days <= 0:
                    return
                progress.console.print(
                    f"skipped {skip_days} days "
                    f"({skip_hours} hours already stored)"
                )
                skip_days = 0
                skip_hours = 0

            def on_day_start(
                *,
                index: int,
                days: int,
                day_start: datetime,
            ) -> None:
                progress.update(
                    task_id,
                    description=f"scanning {_day_label(day_start)}",
                )

            def on_day(
                *,
                index: int,
                days: int,
                day_start: datetime,
                counts: StoreCounts,
                requests: int,
                retries_429: int,
                already: int,
                missing: int,
            ) -> None:
                nonlocal skip_days, skip_hours
                day = _day_label(day_start)
                if missing == 0 and not force:
                    skip_days += 1
                    skip_hours += already
                    progress.update(
                        task_id,
                        completed=index,
                        description=f"scanning {day}",
                    )
                    return
                flush_skips()
                extra = f", {retries_429}×429" if retries_429 else ""
                remaining = days - index
                if index > 0 and remaining > 0:
                    eta = format_eta(
                        (time.monotonic() - started) / index * remaining
                    )
                elif remaining == 0:
                    eta = "0s"
                else:
                    eta = "?"
                if already and not force:
                    summary = (
                        f"stored {counts.events} events, "
                        f"{counts.brackets} brackets  "
                        f"({missing} missing, {already} skipped)"
                    )
                else:
                    summary = (
                        f"stored {counts.events} events, "
                        f"{counts.brackets} brackets"
                    )
                # Permanent day log (not overwritten by the live bar).
                progress.console.print(
                    f"[{index}/{days}] {day}  "
                    f"{summary}  "
                    f"({requests} req{extra})  "
                    f"ETA {eta}"
                )
                progress.update(
                    task_id,
                    completed=index,
                    description=f"finished {day}",
                )

            with httpx.Client(base_url=BASE_URL, timeout=30.0) as http:
                counts = run_backfill(
                    connection,
                    KalshiClient(http, max_rps=rps),
                    start,
                    end,
                    force=force,
                    on_day_start=on_day_start,
                    on_day=on_day,
                    on_skip=on_skip,
                )
            flush_skips()
            progress.update(task_id, description="done")
    finally:
        connection.close()
    console.print(
        f"done  {counts.events} events, {counts.brackets} brackets"
    )


def run_streamlit(db: Path) -> int:
    """Launch the local Streamlit coverage/browse UI for ``db``."""
    script = Path(__file__).resolve().parent / "web_app.py"
    env = os.environ.copy()
    env["BOB_DB"] = str(db.resolve())
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(script),
            "--server.address",
            "127.0.0.1",
            "--browser.gatherUsageStats",
            "false",
        ],
        env=env,
    )


@app.command()
def viz(
    db: Annotated[
        Path,
        typer.Option(help="SQLite path."),
    ] = DEFAULT_DB,
) -> None:
    """Open a local Streamlit UI for coverage and settled-event browse."""
    require_gate()
    if not db.is_file():
        console.print(f"[red]Error:[/red] database not found: {db}")
        raise typer.Exit(code=2)
    raise typer.Exit(code=run_streamlit(db))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
