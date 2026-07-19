# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
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
from rich.table import Table

from bob.db import (
    CandleGapError,
    StoreCounts,
    acknowledge_candle_hour_gap,
    connect,
    event_tickers_in_close_range,
    hours_needing_candles,
    initialize_schema,
    market_candle_inventory,
    store_btc_candles,
    store_settled_events,
)
from bob.gate import require_gate
from bob.kalshi import (
    BASE_URL,
    DEFAULT_MARKET_CANDLE_RPS,
    DEFAULT_MAX_RPS,
    KalshiAPIError,
    KalshiAuthError,
    KalshiClient,
    KalshiCredentialsError,
    KalshiParseError,
    expected_kxbtc_event_tickers,
    load_dotenv,
    require_kalshi_credentials,
)
from bob.market_candles import run_backfill_market_candles
from bob.research import (
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    s16,
)
from bob.research.pnl import score_trades_by_minute
from bob.research.runner import run_all_strategy_pnl
from bob.research.s1 import Side
from bob.research.s12 import DEFAULT_TAU
from bob.research.s13 import DEFAULT_P_STAR
from bob.research.s14 import DEFAULT_Q_STAR
from bob.research.s15 import DEFAULT_DWELL, DEFAULT_MOVE
from bob.research.s16 import DEFAULT_MAX_MOVE

DEFAULT_DB = Path("data/bob.sqlite")

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
)
research_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Offline quote-sim research (named strategies: s1–s16).",
)
app.add_typer(research_app, name="research")
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
        raise typer.BadParameter(f"datetime must include timezone or Z: {value!r}")
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
    now: datetime | None = None,
    on_day_start=None,
    on_day=None,
    on_skip=None,
) -> StoreCounts:
    total_events = 0
    total_brackets = 0
    total_candles = 0
    chunks = list(iter_utc_day_chunks(start, end))
    known = set() if force else event_tickers_in_close_range(connection, start, end)
    current = now if now is not None else datetime.now(timezone.utc)
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
        missing = len(expected)
        event_counts = StoreCounts(events=0, brackets=0)
        req_before = client.requests
        retries_before = client.retries_429
        if not force:
            existing = expected & known
            missing_tickers = expected - known
            already = len(existing)
            missing = len(missing_tickers)
            if missing_tickers:
                only_event_tickers = frozenset(missing_tickers)
        if force or missing > 0:
            events = client.fetch_settled_kxbtc(
                start=chunk_start,
                end=chunk_end,
                on_skip=on_skip,
                only_event_tickers=only_event_tickers,
            )
            event_counts = store_settled_events(connection, events)
            if not force:
                known.update(item.event.event_ticker for item in events)
            total_events += event_counts.events
            total_brackets += event_counts.brackets

        candle_hours = hours_needing_candles(
            connection,
            chunk_start,
            chunk_end,
            force=force,
            now=current,
        )
        candle_count = 0
        if candle_hours:
            bars = client.fetch_brti_minute_bars(
                start=chunk_start,
                end=chunk_end,
                hour_starts=candle_hours,
            )
            candle_count = store_btc_candles(connection, bars)
            total_candles += candle_count

        if on_day is not None:
            on_day(
                index=index,
                days=len(chunks),
                day_start=chunk_start,
                counts=StoreCounts(
                    events=event_counts.events,
                    brackets=event_counts.brackets,
                    candles=candle_count,
                ),
                requests=client.requests - req_before,
                retries_429=client.retries_429 - retries_before,
                already=already,
                missing=missing,
                candle_hours=len(candle_hours),
            )
    return StoreCounts(
        events=total_events,
        brackets=total_brackets,
        candles=total_candles,
    )


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
            help="UTC start of close_time / candle range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ],
    end: Annotated[
        datetime,
        typer.Option(
            ...,
            help="UTC end of close_time / candle range [start, end).",
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
                "Max HTTP requests/sec (CF passthrough ≈ 4 on Basic). "
                "Use 0 to disable pacing."
            ),
        ),
    ] = DEFAULT_MAX_RPS,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Refetch events and candles even when already stored.",
        ),
    ] = False,
) -> None:
    """Backfill settled KXBTC events and BRTI 1m candles into SQLite."""
    require_gate()
    load_dotenv()
    if start >= end:
        console.print("[red]Error:[/red] --start must be earlier than --end")
        raise typer.Exit(code=2)
    if rps < 0:
        console.print("[red]Error:[/red] --rps must be >= 0")
        raise typer.Exit(code=2)
    try:
        credentials = require_kalshi_credentials()
    except KalshiCredentialsError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

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
                    f"skipped {skip_days} days ({skip_hours} hours already stored)"
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
                candle_hours: int,
            ) -> None:
                nonlocal skip_days, skip_hours
                day = _day_label(day_start)
                if missing == 0 and candle_hours == 0 and not force:
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
                    eta = format_eta((time.monotonic() - started) / index * remaining)
                elif remaining == 0:
                    eta = "0s"
                else:
                    eta = "?"
                if already and not force:
                    summary = (
                        f"stored {counts.events} events, "
                        f"{counts.brackets} brackets, "
                        f"{counts.candles} candles  "
                        f"({missing} missing, {already} skipped)"
                    )
                else:
                    summary = (
                        f"stored {counts.events} events, "
                        f"{counts.brackets} brackets, "
                        f"{counts.candles} candles"
                    )
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

            try:
                with httpx.Client(base_url=credentials.base_url, timeout=30.0) as http:
                    counts = run_backfill(
                        connection,
                        KalshiClient(
                            http,
                            credentials=credentials,
                            max_rps=rps,
                        ),
                        start,
                        end,
                        force=force,
                        on_day_start=on_day_start,
                        on_day=on_day,
                        on_skip=on_skip,
                    )
            except (KalshiAuthError, KalshiAPIError) as exc:
                flush_skips()
                console.print(f"[red]Error:[/red] {exc}")
                raise typer.Exit(code=1) from exc
            flush_skips()
            progress.update(task_id, description="done")
    finally:
        connection.close()
    console.print(
        f"done  {counts.events} events, {counts.brackets} brackets, "
        f"{counts.candles} candles"
    )


@app.command("backfill-market-candles")
def backfill_market_candles(
    start: Annotated[
        datetime,
        typer.Option(
            ...,
            help="UTC start of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ],
    end: Annotated[
        datetime,
        typer.Option(
            ...,
            help="UTC end of event close_ts range [start, end).",
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
                "Max HTTP requests/sec for market quote candles "
                "(separate from CF/BRTI backfill). Use 0 to disable pacing."
            ),
        ),
    ] = DEFAULT_MARKET_CANDLE_RPS,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Refetch quote hours even when all 60 slots already exist.",
        ),
    ] = False,
) -> None:
    """Backfill 1m YES bid/ask quote candles into market_candles."""
    require_gate()
    load_dotenv()
    if start >= end:
        console.print("[red]Error:[/red] --start must be earlier than --end")
        raise typer.Exit(code=2)
    if rps < 0:
        console.print("[red]Error:[/red] --rps must be >= 0")
        raise typer.Exit(code=2)
    if not db.is_file():
        console.print(f"[red]Error:[/red] database not found: {db}")
        raise typer.Exit(code=2)

    connection = connect(db)
    try:
        initialize_schema(connection)
        inventory = market_candle_inventory(
            connection, start, end, force=force
        )
        console.print(
            "inventory  "
            f"{inventory.events} events, {inventory.markets} markets, "
            f"~{inventory.expected_rows} rows, "
            f"{inventory.needing_fetch} requests to fetch, "
            f"{inventory.already_complete} already complete"
        )
        if inventory.needing_fetch == 0:
            console.print("done  nothing to fetch")
            return

        started = time.monotonic()
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
            task_id = progress.add_task(
                "market quotes", total=inventory.needing_fetch
            )

            def on_progress(index: int, total: int, ticker: str) -> None:
                progress.update(
                    task_id,
                    completed=index,
                    description=ticker,
                )

            try:
                with httpx.Client(base_url=BASE_URL, timeout=30.0) as http:
                    counts = run_backfill_market_candles(
                        connection,
                        KalshiClient(http, max_rps=rps),
                        start,
                        end,
                        force=force,
                        on_progress=on_progress,
                    )
            except (
                KalshiAuthError,
                KalshiAPIError,
                KalshiParseError,
                httpx.HTTPError,
            ) as exc:
                console.print(f"[red]Error:[/red] {exc}")
                raise typer.Exit(code=1) from exc
            progress.update(task_id, description="done")
    finally:
        connection.close()

    elapsed = time.monotonic() - started
    req_rate = counts.fetched / elapsed if elapsed > 0 else 0.0
    console.print(
        f"done  fetched={counts.fetched} written={counts.written} "
        f"skipped={counts.skipped} empty={counts.empty} "
        f"errors={counts.errors}  ({req_rate:.2f} ticker/s)"
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


@app.command("ack-candle-gap")
def ack_candle_gap(
    hour: Annotated[
        datetime,
        typer.Option(
            ...,
            help="UTC hour start to acknowledge (must be hour-aligned).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ],
    db: Annotated[
        Path,
        typer.Option(help="SQLite path."),
    ] = DEFAULT_DB,
) -> None:
    """Acknowledge an upstream CF candle hole for coverage accounting."""
    require_gate()
    if not db.is_file():
        console.print(f"[red]Error:[/red] database not found: {db}")
        raise typer.Exit(code=2)
    connection = connect(db)
    try:
        initialize_schema(connection)
        try:
            acknowledge_candle_hour_gap(connection, hour)
        except CandleGapError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=2) from exc
    finally:
        connection.close()
    console.print(
        f"acknowledged upstream gap at "
        f"{hour.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )


def parse_checkpoint_minutes(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise typer.BadParameter("minutes must be a non-empty comma list")
    minutes: list[int] = []
    for part in parts:
        try:
            minute = int(part)
        except ValueError as exc:
            raise typer.BadParameter(
                f"invalid minute {part!r}; expected integers 1..59"
            ) from exc
        if not 1 <= minute <= 59:
            raise typer.BadParameter(f"minute must be in 1..59, got {minute}")
        minutes.append(minute)
    if len(set(minutes)) != len(minutes):
        raise typer.BadParameter("minutes must be unique")
    return tuple(minutes)


def parse_tau(value: str) -> Decimal:
    try:
        tau = Decimal(value)
    except InvalidOperation as exc:
        raise typer.BadParameter("tau must be a decimal in (0, 1)") from exc
    if not tau.is_finite() or not (Decimal("0") < tau < Decimal("1")):
        raise typer.BadParameter("tau must be a decimal in (0, 1)")
    return tau


def parse_positive_decimal(value: str) -> Decimal:
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise typer.BadParameter("must be a positive decimal") from exc
    if not amount.is_finite() or amount <= 0:
        raise typer.BadParameter("must be a positive decimal")
    return amount


def parse_dwell(value: str) -> int:
    try:
        dwell = int(value)
    except ValueError as exc:
        raise typer.BadParameter("dwell must be an integer >= 2") from exc
    if dwell < 2:
        raise typer.BadParameter("dwell must be an integer >= 2")
    return dwell


def parse_side(value: str) -> Side:
    normalized = value.strip().lower()
    if normalized == "yes":
        return "yes"
    if normalized == "no":
        return "no"
    raise typer.BadParameter("side must be 'yes' or 'no'")


def _print_research_banner() -> None:
    console.print(
        "[dim]quote-sim · 1 contract · YES@ask / NO@(1−bid) · "
        "gross only (no fees) · minute close ≠ proven fill[/dim]"
    )


def _fmt_return(pnl) -> str:
    if pnl.return_on_premium is None:
        return "—"
    return f"{pnl.return_on_premium * 100:.1f}%"


def _fmt_win_rate(pnl) -> str:
    if pnl.win_rate is None:
        return "—"
    return f"{pnl.win_rate * 100:.1f}%"


def _print_research_tables(
    *,
    strategy: str,
    summary: str,
    side: Side,
    by_minute,
    outcome_minutes,
    abstention_attr: str | None = None,
) -> None:
    _print_research_banner()
    table = Table(title=f"{strategy} · {summary} · side={side}")
    table.add_column("minute", justify="right")
    table.add_column("n", justify="right")
    table.add_column("excl", justify="right")
    table.add_column("premium", justify="right")
    table.add_column("gross", justify="right")
    table.add_column("return", justify="right")
    table.add_column("win_rate", justify="right")
    if abstention_attr is not None:
        table.add_column("abstained", justify="right")
    outcome_by_minute = {stats.minute: stats for stats in outcome_minutes}
    for minute, pnl in by_minute:
        row = [
            str(minute),
            str(pnl.quote_eligible),
            str(pnl.quote_excluded),
            f"{pnl.premium:.2f}",
            f"{pnl.gross:.2f}",
            _fmt_return(pnl),
            _fmt_win_rate(pnl),
        ]
        if abstention_attr is not None:
            stats = outcome_by_minute.get(minute)
            row.append("—" if stats is None else str(getattr(stats, abstention_attr)))
        table.add_row(*row)
    console.print(table)

    exclusion_keys = sorted(
        {reason for stats in outcome_minutes for reason in stats.exclusions}
    )
    if exclusion_keys:
        excl = Table(title="Exclusions")
        excl.add_column("minute", justify="right")
        for key in exclusion_keys:
            excl.add_column(key, justify="right")
        for stats in outcome_minutes:
            excl.add_row(
                str(stats.minute),
                *(str(stats.exclusions.get(key, 0)) for key in exclusion_keys),
            )
        console.print(excl)

    if abstention_attr is None:
        return
    abstention_keys = sorted(
        {reason for stats in outcome_minutes for reason in stats.abstentions}
    )
    if abstention_keys:
        abst = Table(title="Abstentions")
        abst.add_column("minute", justify="right")
        for key in abstention_keys:
            abst.add_column(key, justify="right")
        for stats in outcome_minutes:
            abst.add_row(
                str(stats.minute),
                *(str(stats.abstentions.get(key, 0)) for key in abstention_keys),
            )
        console.print(abst)


def _research_common_options(
    db: Path,
    start: datetime | None,
    end: datetime | None,
    minutes: str,
) -> tuple[int, ...]:
    if not db.is_file():
        console.print(f"[red]Error:[/red] database not found: {db}")
        raise typer.Exit(code=2)
    if (start is None) ^ (end is None):
        console.print("[red]Error:[/red] provide both --start and --end, or neither")
        raise typer.Exit(code=2)
    if start is not None and end is not None and start >= end:
        console.print("[red]Error:[/red] --start must be earlier than --end")
        raise typer.Exit(code=2)
    try:
        return parse_checkpoint_minutes(minutes)
    except typer.BadParameter as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _register_research_strategy(
    module,
    *,
    docstring: str,
    minutes_help: str,
    has_abstentions: bool,
) -> None:
    defaults = ",".join(str(minute) for minute in module.DEFAULT_CHECKPOINT_MINUTES)

    def command(
        db: Annotated[
            Path,
            typer.Option(help="SQLite path."),
        ] = DEFAULT_DB,
        minutes: Annotated[
            str,
            typer.Option(help=minutes_help),
        ] = defaults,
        side: Annotated[
            Side,
            typer.Option(
                help="Buy YES or NO on the selected bracket.",
                parser=parse_side,
                metavar="yes|no",
            ),
        ] = "yes",
        start: Annotated[
            datetime | None,
            typer.Option(
                help="UTC start of event close_ts range [start, end).",
                parser=parse_iso_datetime,
                metavar="ISO",
            ),
        ] = None,
        end: Annotated[
            datetime | None,
            typer.Option(
                help="UTC end of event close_ts range [start, end).",
                parser=parse_iso_datetime,
                metavar="ISO",
            ),
        ] = None,
    ) -> None:
        require_gate()
        minute_list = _research_common_options(db, start, end, minutes)
        connection = connect(db)
        try:
            report = module.evaluate(
                connection,
                minutes=minute_list,
                side=side,
                start=start,
                end=end,
            )
            by_minute = score_trades_by_minute(
                connection, report.trades, minute_list
            )
        finally:
            connection.close()
        _print_research_tables(
            strategy=report.strategy,
            summary=module.STRATEGY_SUMMARY,
            side=report.side,
            by_minute=by_minute,
            outcome_minutes=report.minutes,
            abstention_attr="abstained" if has_abstentions else None,
        )

    command.__doc__ = docstring
    command.__name__ = f"research_{module.STRATEGY}"
    research_app.command(module.STRATEGY)(command)


_register_research_strategy(
    s1,
    docstring="s1: current-bracket hold-to-settlement (quote-sim).",
    minutes_help="Comma-separated minutes into the hour (1..59).",
    has_abstentions=False,
)
_register_research_strategy(
    s2,
    docstring="s2: stable-center current-bracket hold-to-settlement (quote-sim).",
    minutes_help=(
        "Comma-separated checkpoint minutes (5..59); uses prior "
        "4 minutes for stability."
    ),
    has_abstentions=True,
)
_register_research_strategy(
    s3,
    docstring="s3: linear-trend projected bracket hold-to-settlement (quote-sim).",
    minutes_help="Comma-separated checkpoint minutes (10..59).",
    has_abstentions=True,
)
_register_research_strategy(
    s4,
    docstring="s4: half-reversion to hourly open hold-to-settlement (quote-sim).",
    minutes_help="Comma-separated checkpoint minutes (1..59).",
    has_abstentions=True,
)
_register_research_strategy(
    s5,
    docstring="s5: volatility-buffered current-bracket hold (quote-sim).",
    minutes_help="Comma-separated checkpoint minutes (11..59).",
    has_abstentions=True,
)
_register_research_strategy(
    s6,
    docstring="s6: directed adjacent-bracket breakout hold (quote-sim).",
    minutes_help="Comma-separated checkpoint minutes (3..59).",
    has_abstentions=True,
)
_register_research_strategy(
    s7,
    docstring="s7: dominant-bracket occupancy hold-to-settlement (quote-sim).",
    minutes_help="Comma-separated checkpoint minutes (1..59).",
    has_abstentions=True,
)
_register_research_strategy(
    s8,
    docstring="s8: horizon-confirmed current-bracket hold (quote-sim).",
    minutes_help="Comma-separated checkpoint minutes (31..59).",
    has_abstentions=True,
)
_register_research_strategy(
    s9,
    docstring="s9: matched-horizon excursion-buffered hold (quote-sim).",
    minutes_help="Comma-separated checkpoint minutes (30..59).",
    has_abstentions=True,
)
_register_research_strategy(
    s10,
    docstring="s10: majority-range-persistent current-bracket hold (quote-sim).",
    minutes_help="Comma-separated checkpoint minutes (30..59).",
    has_abstentions=True,
)
_register_research_strategy(
    s11,
    docstring="s11: matched-horizon path-replay current-bracket hold (quote-sim).",
    minutes_help="Comma-separated checkpoint minutes (31..59).",
    has_abstentions=True,
)


@research_app.command("s12")
def research_s12(
    db: Annotated[
        Path,
        typer.Option(help="SQLite path."),
    ] = DEFAULT_DB,
    minutes: Annotated[
        str,
        typer.Option(help="Comma-separated checkpoint minutes (40..59)."),
    ] = ",".join(str(minute) for minute in s12.DEFAULT_CHECKPOINT_MINUTES),
    side: Annotated[
        Side,
        typer.Option(
            help="Buy YES or NO on the selected bracket.",
            parser=parse_side,
            metavar="yes|no",
        ),
    ] = "yes",
    tau: Annotated[
        Decimal,
        typer.Option(
            help="Trade when estimated print-escape probability ≤ 1-tau.",
            parser=parse_tau,
            metavar="0..1",
        ),
    ] = DEFAULT_TAU,
    start: Annotated[
        datetime | None,
        typer.Option(
            help="UTC start of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ] = None,
    end: Annotated[
        datetime | None,
        typer.Option(
            help="UTC end of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ] = None,
) -> None:
    """s12: calibrated print-risk current-bracket hold (quote-sim)."""
    require_gate()
    minute_list = _research_common_options(db, start, end, minutes)
    connection = connect(db)
    try:
        report = s12.evaluate(
            connection,
            minutes=minute_list,
            side=side,
            tau=tau,
            start=start,
            end=end,
        )
        by_minute = score_trades_by_minute(connection, report.trades, minute_list)
    finally:
        connection.close()
    _print_research_tables(
        strategy=report.strategy,
        summary=f"{s12.STRATEGY_SUMMARY} · tau={report.tau}",
        side=report.side,
        by_minute=by_minute,
        outcome_minutes=report.minutes,
        abstention_attr="abstained",
    )


@research_app.command("s13")
def research_s13(
    db: Annotated[
        Path,
        typer.Option(help="SQLite path."),
    ] = DEFAULT_DB,
    minutes: Annotated[
        str,
        typer.Option(help="Comma-separated checkpoint minutes (30..59)."),
    ] = ",".join(str(minute) for minute in s13.DEFAULT_CHECKPOINT_MINUTES),
    side: Annotated[
        Side,
        typer.Option(
            help="Buy YES or NO on the selected bracket.",
            parser=parse_side,
            metavar="yes|no",
        ),
    ] = "yes",
    p_star: Annotated[
        Decimal,
        typer.Option(
            "--p-star",
            help="Trade when estimated terminal win probability ≥ p-star.",
            parser=parse_tau,
            metavar="0..1",
        ),
    ] = DEFAULT_P_STAR,
    start: Annotated[
        datetime | None,
        typer.Option(
            help="UTC start of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ] = None,
    end: Annotated[
        datetime | None,
        typer.Option(
            help="UTC end of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ] = None,
) -> None:
    """s13: regime-pooled terminal-probability current-bracket hold (quote-sim)."""
    require_gate()
    minute_list = _research_common_options(db, start, end, minutes)
    connection = connect(db)
    try:
        report = s13.evaluate(
            connection,
            minutes=minute_list,
            side=side,
            p_star=p_star,
            start=start,
            end=end,
        )
        by_minute = score_trades_by_minute(connection, report.trades, minute_list)
    finally:
        connection.close()
    _print_research_tables(
        strategy=report.strategy,
        summary=f"{s13.STRATEGY_SUMMARY} · p_star={report.p_star}",
        side=report.side,
        by_minute=by_minute,
        outcome_minutes=report.minutes,
        abstention_attr="abstained",
    )


@research_app.command("s14")
def research_s14(
    db: Annotated[
        Path,
        typer.Option(help="SQLite path."),
    ] = DEFAULT_DB,
    minutes: Annotated[
        str,
        typer.Option(help="Comma-separated checkpoint minutes (35..59)."),
    ] = ",".join(str(minute) for minute in s14.DEFAULT_CHECKPOINT_MINUTES),
    side: Annotated[
        Side,
        typer.Option(
            help="Buy YES or NO on the selected bracket.",
            parser=parse_side,
            metavar="yes|no",
        ),
    ] = "yes",
    p_star: Annotated[
        Decimal,
        typer.Option(
            "--p-star",
            help="Trade when recent-clock terminal p̂ ≥ p-star.",
            parser=parse_tau,
            metavar="0..1",
        ),
    ] = DEFAULT_P_STAR,
    q_star: Annotated[
        Decimal,
        typer.Option(
            "--q-star",
            help="Abstain when whole-hour-clock terminal p̂ > q-star.",
            parser=parse_tau,
            metavar="0..1",
        ),
    ] = DEFAULT_Q_STAR,
    start: Annotated[
        datetime | None,
        typer.Option(
            help="UTC start of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ] = None,
    end: Annotated[
        datetime | None,
        typer.Option(
            help="UTC end of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ] = None,
) -> None:
    """s14: two-clock vol-disagreement current-bracket hold (quote-sim)."""
    require_gate()
    minute_list = _research_common_options(db, start, end, minutes)
    connection = connect(db)
    try:
        report = s14.evaluate(
            connection,
            minutes=minute_list,
            side=side,
            p_star=p_star,
            q_star=q_star,
            start=start,
            end=end,
        )
        by_minute = score_trades_by_minute(connection, report.trades, minute_list)
    finally:
        connection.close()
    _print_research_tables(
        strategy=report.strategy,
        summary=(
            f"{s14.STRATEGY_SUMMARY} · p_star={report.p_star} · "
            f"q_star={report.q_star}"
        ),
        side=report.side,
        by_minute=by_minute,
        outcome_minutes=report.minutes,
        abstention_attr="abstained",
    )


@research_app.command("s15")
def research_s15(
    db: Annotated[
        Path,
        typer.Option(help="SQLite path."),
    ] = DEFAULT_DB,
    minutes: Annotated[
        str,
        typer.Option(help="Comma-separated checkpoint minutes (35..59)."),
    ] = ",".join(str(minute) for minute in s15.DEFAULT_CHECKPOINT_MINUTES),
    side: Annotated[
        Side,
        typer.Option(
            help="Buy YES or NO on the selected bracket.",
            parser=parse_side,
            metavar="yes|no",
        ),
    ] = "yes",
    move: Annotated[
        Decimal,
        typer.Option(
            help="Minimum |close_M − open_1| impulse in dollars.",
            parser=parse_positive_decimal,
            metavar="USD",
        ),
    ] = DEFAULT_MOVE,
    dwell: Annotated[
        int,
        typer.Option(
            help="Consecutive closes that must stay in the current bracket.",
            parser=parse_dwell,
            metavar="N",
        ),
    ] = DEFAULT_DWELL,
    start: Annotated[
        datetime | None,
        typer.Option(
            help="UTC start of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ] = None,
    end: Annotated[
        datetime | None,
        typer.Option(
            help="UTC end of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ] = None,
) -> None:
    """s15: impulse–flag arrival current-bracket hold (quote-sim)."""
    require_gate()
    minute_list = _research_common_options(db, start, end, minutes)
    connection = connect(db)
    try:
        report = s15.evaluate(
            connection,
            minutes=minute_list,
            side=side,
            move=move,
            dwell=dwell,
            start=start,
            end=end,
        )
        by_minute = score_trades_by_minute(connection, report.trades, minute_list)
    finally:
        connection.close()
    _print_research_tables(
        strategy=report.strategy,
        summary=(
            f"{s15.STRATEGY_SUMMARY} · move={report.move} · dwell={report.dwell}"
        ),
        side=report.side,
        by_minute=by_minute,
        outcome_minutes=report.minutes,
        abstention_attr="abstained",
    )


@research_app.command("s16")
def research_s16(
    db: Annotated[
        Path,
        typer.Option(help="SQLite path."),
    ] = DEFAULT_DB,
    minutes: Annotated[
        str,
        typer.Option(help="Comma-separated checkpoint minutes (31..59)."),
    ] = ",".join(str(minute) for minute in s16.DEFAULT_CHECKPOINT_MINUTES),
    side: Annotated[
        Side,
        typer.Option(
            help="Buy YES or NO on the selected bracket.",
            parser=parse_side,
            metavar="yes|no",
        ),
    ] = "yes",
    max_move: Annotated[
        Decimal,
        typer.Option(
            "--max-move",
            help="Abstain when |close_M − open_1| ≥ max-move (dollars).",
            parser=parse_positive_decimal,
            metavar="USD",
        ),
    ] = DEFAULT_MAX_MOVE,
    start: Annotated[
        datetime | None,
        typer.Option(
            help="UTC start of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ] = None,
    end: Annotated[
        datetime | None,
        typer.Option(
            help="UTC end of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ] = None,
) -> None:
    """s16: horizon-confirmed calm current-bracket hold (quote-sim)."""
    require_gate()
    minute_list = _research_common_options(db, start, end, minutes)
    connection = connect(db)
    try:
        report = s16.evaluate(
            connection,
            minutes=minute_list,
            side=side,
            max_move=max_move,
            start=start,
            end=end,
        )
        by_minute = score_trades_by_minute(connection, report.trades, minute_list)
    finally:
        connection.close()
    _print_research_tables(
        strategy=report.strategy,
        summary=f"{s16.STRATEGY_SUMMARY} · max_move={report.max_move}",
        side=report.side,
        by_minute=by_minute,
        outcome_minutes=report.minutes,
        abstention_attr="abstained",
    )


def _print_all_research_table(summaries) -> None:
    minutes = []
    for item in summaries:
        for minute, _pnl in item.by_minute:
            if minute not in minutes:
                minutes.append(minute)
    if not minutes:
        table = Table(title="research all")
        table.add_column("strategy")
        console.print(table)
        return
    for minute in minutes:
        table = Table(title=f"research all · M{minute}")
        table.add_column("strategy")
        table.add_column("n", justify="right")
        table.add_column("excl", justify="right")
        table.add_column("premium", justify="right")
        table.add_column("gross", justify="right")
        table.add_column("return", justify="right")
        table.add_column("win_rate", justify="right")
        for item in summaries:
            pnl = next(
                (report for m, report in item.by_minute if m == minute),
                None,
            )
            if pnl is None:
                continue
            table.add_row(
                item.strategy,
                str(pnl.quote_eligible),
                str(pnl.quote_excluded),
                f"{pnl.premium:.2f}",
                f"{pnl.gross:.2f}",
                _fmt_return(pnl),
                _fmt_win_rate(pnl),
            )
        console.print(table)


def _validate_minutes_for_all(minutes: tuple[int, ...]) -> None:
    """s12 requires ≥40; shared --minutes for all strategies must clear that floor."""
    for minute in minutes:
        if not 40 <= minute <= 59:
            console.print(
                "[red]Error:[/red] for research all, "
                "minutes must be in 40..59 (s12 floor)"
            )
            raise typer.Exit(code=2)


@research_app.command("all")
def research_all(
    db: Annotated[
        Path,
        typer.Option(help="SQLite path."),
    ] = DEFAULT_DB,
    minutes: Annotated[
        str,
        typer.Option(help="Comma-separated checkpoint minutes (40..59)."),
    ] = "50",
    side: Annotated[
        Side,
        typer.Option(
            help="Buy YES or NO on the selected bracket.",
            parser=parse_side,
            metavar="yes|no",
        ),
    ] = "yes",
    tau: Annotated[
        Decimal,
        typer.Option(
            help="s12 only: trade when escape probability ≤ 1-tau.",
            parser=parse_tau,
            metavar="0..1",
        ),
    ] = DEFAULT_TAU,
    p_star: Annotated[
        Decimal,
        typer.Option(
            "--p-star",
            help="s13/s14: quality bar on terminal p̂ (recent clock for s14).",
            parser=parse_tau,
            metavar="0..1",
        ),
    ] = DEFAULT_P_STAR,
    q_star: Annotated[
        Decimal,
        typer.Option(
            "--q-star",
            help="s14 only: abstain when whole-hour-clock p̂ > q-star.",
            parser=parse_tau,
            metavar="0..1",
        ),
    ] = DEFAULT_Q_STAR,
    move: Annotated[
        Decimal,
        typer.Option(
            help="s15 only: minimum |close_M − open_1| impulse in dollars.",
            parser=parse_positive_decimal,
            metavar="USD",
        ),
    ] = DEFAULT_MOVE,
    dwell: Annotated[
        int,
        typer.Option(
            help="s15 only: consecutive closes in the current bracket.",
            parser=parse_dwell,
            metavar="N",
        ),
    ] = DEFAULT_DWELL,
    max_move: Annotated[
        Decimal,
        typer.Option(
            "--max-move",
            help="s16 only: abstain when |close_M − open_1| ≥ max-move.",
            parser=parse_positive_decimal,
            metavar="USD",
        ),
    ] = DEFAULT_MAX_MOVE,
    start: Annotated[
        datetime | None,
        typer.Option(
            help="UTC start of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ] = None,
    end: Annotated[
        datetime | None,
        typer.Option(
            help="UTC end of event close_ts range [start, end).",
            parser=parse_iso_datetime,
            metavar="ISO",
        ),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(help="Process pool size (default: min(16, CPUs))."),
    ] = None,
) -> None:
    """Run s1–s16 quote-sim in parallel; one table per checkpoint minute."""
    require_gate()
    minute_list = _research_common_options(db, start, end, minutes)
    _validate_minutes_for_all(minute_list)
    if workers is not None and workers < 1:
        console.print("[red]Error:[/red] --workers must be >= 1")
        raise typer.Exit(code=2)
    _print_research_banner()
    try:
        summaries = run_all_strategy_pnl(
            db,
            minutes=minute_list,
            side=side,
            start=start,
            end=end,
            tau=tau,
            p_star=p_star,
            q_star=q_star,
            move=move,
            dwell=dwell,
            max_move=max_move,
            workers=workers,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _print_all_research_table(summaries)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
