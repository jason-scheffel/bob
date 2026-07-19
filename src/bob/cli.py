# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import httpx
import typer

from bob.db import StoreCounts, connect, initialize_schema, store_settled_events
from bob.gate import require_gate
from bob.kalshi import BASE_URL, KalshiClient

DEFAULT_DB = Path("data/bob.sqlite")

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
)


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


def run_backfill(
    connection,
    client: KalshiClient,
    start: datetime,
    end: datetime,
) -> StoreCounts:
    events = client.fetch_settled_kxbtc(start=start, end=end)
    return store_settled_events(connection, events)


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
) -> None:
    """Backfill settled KXBTC events and brackets into SQLite."""
    require_gate()
    if start >= end:
        typer.echo("Error: --start must be earlier than --end", err=True)
        raise typer.Exit(code=2)

    connection = connect(db)
    try:
        initialize_schema(connection)
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as http:
            counts = run_backfill(
                connection,
                KalshiClient(http),
                start,
                end,
            )
    finally:
        connection.close()
    typer.echo(f"stored {counts.events} events, {counts.brackets} brackets")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
