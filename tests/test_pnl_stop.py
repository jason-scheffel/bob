# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bob.cli import app
from bob.db import (
    MarketQuoteBar,
    connect,
    initialize_schema,
    store_btc_candles,
    store_market_candles,
    store_settled_events,
)
from bob.research.common import checkpoint_end_ts
from bob.research.pnl import score_trades, score_trades_by_minute
from bob.research.trades import TradeObservation
from helpers import RESEARCH_CLOSE, research_flat_bars, research_settled

CLOSE = datetime(2099, 6, 1, 12, 0, tzinfo=timezone.utc)
EVENT = "KXBTC-99JUN0108"
TICKER = f"{EVENT}-A"
# Obviously synthetic ids for dense overlay matrix (not live product tickers).
SYNTH_EVENT = "SYNTH-EVENT-PNL-001"
SYNTH_TICKER = f"{SYNTH_EVENT}-BAND-A"
runner = CliRunner()


def _obs(*, minute: int = 45, won: bool = True, side: str = "yes") -> TradeObservation:
    return TradeObservation(
        event_ticker=EVENT,
        market_ticker=TICKER,
        minute=minute,
        end_ts=checkpoint_end_ts(CLOSE, minute),
        side=side,  # type: ignore[arg-type]
        won=won,
    )


def _synth_obs(
    *,
    minute: int = 45,
    won: bool = True,
    side: str = "yes",
    event: str = SYNTH_EVENT,
    ticker: str | None = None,
) -> TradeObservation:
    market = ticker if ticker is not None else f"{event}-BAND-A"
    return TradeObservation(
        event_ticker=event,
        market_ticker=market,
        minute=minute,
        end_ts=checkpoint_end_ts(CLOSE, minute),
        side=side,  # type: ignore[arg-type]
        won=won,
    )


def _quote(minute: int, *, bid: str, ask: str) -> MarketQuoteBar:
    return MarketQuoteBar(
        ticker=TICKER,
        end_ts=checkpoint_end_ts(CLOSE, minute),
        yes_bid_close=bid,
        yes_ask_close=ask,
    )


def _synth_quote(
    minute: int,
    *,
    bid: str,
    ask: str,
    ticker: str = SYNTH_TICKER,
) -> MarketQuoteBar:
    return MarketQuoteBar(
        ticker=ticker,
        end_ts=checkpoint_end_ts(CLOSE, minute),
        yes_bid_close=bid,
        yes_ask_close=ask,
    )


def test_no_stop_bid_matches_hold_to_settlement() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.50", ask="0.55"),
            _quote(55, bid="0.20", ask="0.25"),
        ],
    )
    obs = _obs(minute=45, won=True)
    plain = score_trades(connection, [obs])
    with_from = score_trades(
        connection, [obs], stop_bid=None, stop_from=55
    )
    assert plain.gross == with_from.gross == Decimal("0.45")
    assert plain.stopped == with_from.stopped == 0
    assert plain.trades[0].stopped is False
    connection.close()


def test_stop_exits_at_first_breach_from_stop_from() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.50", ask="0.55"),
            _quote(50, bid="0.20", ask="0.25"),
            _quote(55, bid="0.28", ask="0.32"),
            _quote(56, bid="0.25", ask="0.30"),
        ],
    )
    obs = _obs(minute=45, won=True)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        stop_from=55,
    )
    trade = pnl.trades[0]
    assert trade.stopped is True
    assert trade.exit_minute == 55
    assert trade.settlement == Decimal("0.28")
    assert trade.gross == Decimal("0.28") - Decimal("0.55")
    assert pnl.stopped == 1
    assert pnl.wins == 0
    assert pnl.losses == 1
    connection.close()


def test_breach_before_stop_from_is_ignored() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.50", ask="0.55"),
            _quote(50, bid="0.10", ask="0.15"),
            _quote(56, bid="0.60", ask="0.65"),
        ],
    )
    obs = _obs(minute=45, won=True)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        stop_from=55,
    )
    assert pnl.trades[0].stopped is False
    assert pnl.trades[0].settlement == Decimal("1")
    assert pnl.gross == Decimal("0.45")
    connection.close()


def test_sparse_bars_skipped_until_breach() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.50", ask="0.55"),
            MarketQuoteBar(
                ticker=TICKER,
                end_ts=checkpoint_end_ts(CLOSE, 55),
                yes_bid_close=None,
                yes_ask_close=None,
            ),
            _quote(57, bid="0.22", ask="0.24"),
        ],
    )
    obs = _obs(minute=45, won=False)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        stop_from=55,
    )
    trade = pnl.trades[0]
    assert trade.stopped is True
    assert trade.exit_minute == 57
    assert trade.settlement == Decimal("0.22")
    connection.close()


def test_no_breach_holds_to_settlement() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.50", ask="0.55"),
            _quote(55, bid="0.40", ask="0.45"),
            _quote(59, bid="0.35", ask="0.40"),
        ],
    )
    obs = _obs(minute=45, won=False)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        stop_from=55,
    )
    assert pnl.trades[0].stopped is False
    assert pnl.trades[0].settlement == Decimal("0")
    assert pnl.gross == Decimal("-0.55")
    connection.close()


def test_no_side_stops_on_one_minus_ask() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.40", ask="0.45"),
            # NO mark = 1 - ask; ask 0.75 → mark 0.25 ≤ 0.30
            _quote(55, bid="0.70", ask="0.75"),
        ],
    )
    obs = _obs(minute=45, won=True, side="no")
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        stop_from=55,
    )
    trade = pnl.trades[0]
    assert trade.premium == Decimal("0.60")
    assert trade.stopped is True
    assert trade.settlement == Decimal("0.25")
    connection.close()


def test_take_pct_exits_on_return_on_premium() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.48", ask="0.50"),
            # take level = 0.50 * 1.20 = 0.60
            _quote(55, bid="0.62", ask="0.65"),
        ],
    )
    obs = _obs(minute=45, won=False)
    pnl = score_trades(
        connection,
        [obs],
        take_pct=Decimal("0.20"),
        stop_from=55,
    )
    trade = pnl.trades[0]
    assert trade.taken is True
    assert trade.stopped is False
    assert trade.exit_minute == 55
    assert trade.settlement == Decimal("0.62")
    assert trade.gross == Decimal("0.12")
    assert pnl.taken == 1
    assert pnl.wins == 1
    connection.close()


def test_first_touch_stop_before_later_take() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.48", ask="0.50"),
            _quote(55, bid="0.28", ask="0.32"),
            _quote(56, bid="0.70", ask="0.75"),
        ],
    )
    obs = _obs(minute=45, won=True)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        take_pct=Decimal("0.20"),
        stop_from=55,
    )
    trade = pnl.trades[0]
    assert trade.stopped is True
    assert trade.taken is False
    assert trade.exit_minute == 55
    assert trade.settlement == Decimal("0.28")
    connection.close()


def test_same_bar_prefers_stop_over_take() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _quote(45, bid="0.48", ask="0.50"),
            # take level 0.60; stop_bid 0.70 → mark 0.65 hits both
            _quote(55, bid="0.65", ask="0.68"),
        ],
    )
    obs = _obs(minute=45, won=True)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.70"),
        take_pct=Decimal("0.20"),
        stop_from=55,
    )
    trade = pnl.trades[0]
    assert trade.stopped is True
    assert trade.taken is False
    assert trade.settlement == Decimal("0.65")
    connection.close()


def test_invalid_stop_params_raise() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    with pytest.raises(ValueError, match="stop_bid"):
        score_trades(connection, [], stop_bid=Decimal("1.5"))
    with pytest.raises(ValueError, match="stop_from"):
        score_trades(
            connection, [], stop_bid=Decimal("0.3"), stop_from=0
        )
    with pytest.raises(ValueError, match="take_pct"):
        score_trades(connection, [], take_pct=Decimal("0"))
    connection.close()


def test_research_s21_cli_accepts_stop_and_take(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    store_btc_candles(connection, research_flat_bars(range(1, 56), "150"))
    store_market_candles(
        connection,
        [
            MarketQuoteBar(
                ticker="KXBTC-99JUN0108-A",
                end_ts=checkpoint_end_ts(RESEARCH_CLOSE, 55),
                yes_bid_close="0.50",
                yes_ask_close="0.55",
            )
        ],
    )
    connection.close()

    result = runner.invoke(
        app,
        [
            "research",
            "s21",
            "--db",
            str(db_path),
            "--minutes",
            "55",
            "--stop-bid",
            "0.30",
            "--take-pct",
            "0.20",
        ],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert "stop-bid≤0.30" in result.output
    assert "take-pct=0.20" in result.output
    assert "first touch" in result.output
    assert "taken" in result.output


def test_no_side_take_uses_one_minus_ask() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _synth_quote(45, bid="0.40", ask="0.45"),
            # premium NO = 1-0.40 = 0.60; take @ 0.60*1.20=0.72
            # mark NO = 1-ask; ask 0.20 → mark 0.80 ≥ 0.72
            _synth_quote(55, bid="0.15", ask="0.20"),
        ],
    )
    obs = _synth_obs(minute=45, won=False, side="no")
    pnl = score_trades(
        connection,
        [obs],
        take_pct=Decimal("0.20"),
        stop_from=55,
    )
    trade = pnl.trades[0]
    assert trade.premium == Decimal("0.60")
    assert trade.taken is True
    assert trade.settlement == Decimal("0.80")
    assert pnl.wins == 1
    connection.close()


@pytest.mark.parametrize(
    ("bid", "stop_bid", "should_stop"),
    [
        ("0.30", "0.30", True),
        ("0.31", "0.30", False),
        ("0.29", "0.30", True),
    ],
)
def test_stop_exact_equality(bid: str, stop_bid: str, should_stop: bool) -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _synth_quote(45, bid="0.50", ask="0.55"),
            _synth_quote(55, bid=bid, ask="0.40"),
        ],
    )
    obs = _synth_obs(minute=45, won=True)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal(stop_bid),
        stop_from=55,
    )
    assert pnl.trades[0].stopped is should_stop
    connection.close()


@pytest.mark.parametrize(
    ("bid", "should_take"),
    [
        ("0.60", True),  # premium 0.50 * 1.20 = 0.60
        ("0.59", False),
        ("0.61", True),
    ],
)
def test_take_exact_equality(bid: str, should_take: bool) -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _synth_quote(45, bid="0.48", ask="0.50"),
            _synth_quote(55, bid=bid, ask="0.70"),
        ],
    )
    obs = _synth_obs(minute=45, won=False)
    pnl = score_trades(
        connection,
        [obs],
        take_pct=Decimal("0.20"),
        stop_from=55,
    )
    assert pnl.trades[0].taken is should_take
    connection.close()


def test_entry_after_stop_from_skips_entry_bar() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _synth_quote(57, bid="0.10", ask="0.12"),  # entry bar — must ignore
            _synth_quote(58, bid="0.25", ask="0.28"),  # first eligible
        ],
    )
    obs = _synth_obs(minute=57, won=True)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        stop_from=55,
    )
    trade = pnl.trades[0]
    assert trade.stopped is True
    assert trade.exit_minute == 58
    assert trade.settlement == Decimal("0.25")
    connection.close()


def test_stop_from_59_with_no_later_bar_holds() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _synth_quote(55, bid="0.50", ask="0.55"),
            _synth_quote(59, bid="0.10", ask="0.12"),  # entry bar when stop_from=59
        ],
    )
    obs = _synth_obs(minute=59, won=True)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        stop_from=59,
    )
    trade = pnl.trades[0]
    assert trade.stopped is False
    assert trade.settlement == Decimal("1")
    connection.close()


def test_take_before_later_stop() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _synth_quote(45, bid="0.48", ask="0.50"),
            _synth_quote(55, bid="0.62", ask="0.65"),  # take first
            _synth_quote(56, bid="0.10", ask="0.12"),  # would stop later
        ],
    )
    obs = _synth_obs(minute=45, won=False)
    pnl = score_trades(
        connection,
        [obs],
        stop_bid=Decimal("0.30"),
        take_pct=Decimal("0.20"),
        stop_from=55,
    )
    trade = pnl.trades[0]
    assert trade.taken is True
    assert trade.stopped is False
    assert trade.exit_minute == 55
    connection.close()


def test_mixed_report_aggregates_every_field() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    t_stop = f"{SYNTH_EVENT}-STOP"
    t_take = f"{SYNTH_EVENT}-TAKE"
    t_win = f"{SYNTH_EVENT}-HOLDW"
    t_loss = f"{SYNTH_EVENT}-HOLDL"
    store_market_candles(
        connection,
        [
            _synth_quote(45, bid="0.50", ask="0.55", ticker=t_stop),
            _synth_quote(55, bid="0.20", ask="0.22", ticker=t_stop),
            _synth_quote(45, bid="0.48", ask="0.50", ticker=t_take),
            _synth_quote(55, bid="0.65", ask="0.70", ticker=t_take),
            _synth_quote(45, bid="0.40", ask="0.45", ticker=t_win),
            _synth_quote(45, bid="0.40", ask="0.45", ticker=t_loss),
            # invalid entry for exclusion path: missing ask
            MarketQuoteBar(
                ticker=f"{SYNTH_EVENT}-BAD",
                end_ts=checkpoint_end_ts(CLOSE, 45),
                yes_bid_close="0.40",
                yes_ask_close=None,
            ),
        ],
    )
    observations = [
        _synth_obs(minute=45, won=True, ticker=t_stop),
        _synth_obs(minute=45, won=False, ticker=t_take),
        _synth_obs(minute=45, won=True, ticker=t_win),
        _synth_obs(minute=45, won=False, ticker=t_loss),
        _synth_obs(minute=45, won=True, ticker=f"{SYNTH_EVENT}-BAD"),
    ]
    pnl = score_trades(
        connection,
        observations,
        stop_bid=Decimal("0.30"),
        take_pct=Decimal("0.20"),
        stop_from=55,
    )
    assert pnl.strategy_eligible == 5
    assert pnl.quote_excluded == 1
    assert pnl.quote_eligible == 4
    assert pnl.stopped == 1
    assert pnl.taken == 1
    assert pnl.wins == 2  # taken + hold win
    assert pnl.losses == 2  # stopped + hold loss
    assert pnl.premium == (
        Decimal("0.55") + Decimal("0.50") + Decimal("0.45") + Decimal("0.45")
    )
    assert pnl.payout == (
        Decimal("0.20") + Decimal("0.65") + Decimal("1") + Decimal("0")
    )
    assert pnl.gross == pnl.payout - pnl.premium
    kinds = {(t.stopped, t.taken, t.observation.won) for t in pnl.trades}
    assert (True, False, True) in kinds
    assert (False, True, False) in kinds
    assert (False, False, True) in kinds
    assert (False, False, False) in kinds
    connection.close()


def test_score_trades_by_minute_applies_overlays_per_bucket() -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    store_market_candles(
        connection,
        [
            _synth_quote(45, bid="0.50", ask="0.55"),
            _synth_quote(50, bid="0.50", ask="0.55"),
            _synth_quote(55, bid="0.20", ask="0.22"),
            _synth_quote(56, bid="0.20", ask="0.22"),
        ],
    )
    observations = [
        _synth_obs(minute=45, won=True),
        _synth_obs(minute=50, won=True),
    ]
    by_minute = score_trades_by_minute(
        connection,
        observations,
        (45, 50),
        stop_bid=Decimal("0.30"),
        stop_from=55,
    )
    assert [m for m, _ in by_minute] == [45, 50]
    for _minute, report in by_minute:
        assert report.stopped == 1
        assert report.trades[0].exit_minute in {55, 56}
    connection.close()


@pytest.mark.parametrize(
    ("stop_bid", "take_pct", "stop_from", "match"),
    [
        (Decimal("-0.01"), None, 55, "stop_bid"),
        (Decimal("nan"), None, 55, "stop_bid"),
        (None, Decimal("-0.1"), 55, "take_pct"),
        (Decimal("0.3"), None, 60, "stop_from"),
        (Decimal("0.3"), None, -1, "stop_from"),
    ],
)
def test_overlay_validation_matrix(
    stop_bid: Decimal | None,
    take_pct: Decimal | None,
    stop_from: int,
    match: str,
) -> None:
    connection = connect(":memory:")
    initialize_schema(connection)
    with pytest.raises(ValueError, match=match):
        score_trades(
            connection,
            [],
            stop_bid=stop_bid,
            take_pct=take_pct,
            stop_from=stop_from,
        )
    connection.close()


def test_research_cli_reports_stopped_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("bob.cli.require_gate", lambda: None)
    db_path = tmp_path / "bob.sqlite"
    connection = connect(db_path)
    initialize_schema(connection)
    store_settled_events(connection, [research_settled(expiration="150", winner="a")])
    store_btc_candles(connection, research_flat_bars(range(1, 56), "150"))
    store_market_candles(
        connection,
        [
            MarketQuoteBar(
                ticker="KXBTC-99JUN0108-A",
                end_ts=checkpoint_end_ts(RESEARCH_CLOSE, 55),
                yes_bid_close="0.50",
                yes_ask_close="0.55",
            ),
            MarketQuoteBar(
                ticker="KXBTC-99JUN0108-A",
                end_ts=checkpoint_end_ts(RESEARCH_CLOSE, 56),
                yes_bid_close="0.20",
                yes_ask_close="0.22",
            ),
        ],
    )
    connection.close()

    result = runner.invoke(
        app,
        [
            "research",
            "s1",
            "--db",
            str(db_path),
            "--minutes",
            "55",
            "--stop-bid",
            "0.30",
            "--stop-from",
            "55",
        ],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert "stopped" in result.output.lower()
    # display should reflect the early exit
    assert "1" in result.output
