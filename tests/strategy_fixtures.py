# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Obviously synthetic research fixtures (not BTC / not Kalshi product IDs).

Toy units: brackets 100–199.99 / 200–299.99 / 300–399.99 are dimensionless.
Event ids look like SYNTH-EVENT-… so they cannot be mistaken for live series.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from bob.db import MinuteBar, store_btc_candles, store_settled_events
from bob.kalshi import STATUS_COMPLETE, Bracket, Event, SettledEvent
from bob.research.common import checkpoint_end_ts
from bob.research.s1 import Side

# Far-future UTC epoch so timestamps never look like production history.
SYNTH_EPOCH = datetime(2099, 7, 1, 0, 0, tzinfo=timezone.utc)

SeedFn = Callable[[Any], None]


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One evaluate_strategy input; expected output lives in golden JSON."""

    case_id: str
    strategy: str
    minutes: tuple[int, ...]
    side: Side
    seed: SeedFn
    eval_kwargs: Mapping[str, Any] | None = None


def synth_close(slot: int) -> datetime:
    """Distinct hour per slot so global candle keys never collide."""
    return SYNTH_EPOCH + timedelta(hours=slot)


def synth_event_ticker(label: str) -> str:
    return f"SYNTH-EVENT-{label}"


def synthetic_settled(
    *,
    label: str,
    close: datetime,
    expiration: str,
    winner: str,
    open_high: bool = False,
) -> SettledEvent:
    """Three toy bands A/B/C. winner in {'a','b','c'}."""
    event_ticker = synth_event_ticker(label)
    if open_high:
        brackets = (
            Bracket(
                ticker=f"{event_ticker}-BAND-A",
                event_ticker=event_ticker,
                floor_strike=Decimal("100"),
                cap_strike=Decimal("199.99"),
                won=winner == "a",
            ),
            Bracket(
                ticker=f"{event_ticker}-BAND-HIGH",
                event_ticker=event_ticker,
                floor_strike=Decimal("200"),
                cap_strike=None,
                won=winner == "b",
            ),
        )
    else:
        brackets = (
            Bracket(
                ticker=f"{event_ticker}-BAND-A",
                event_ticker=event_ticker,
                floor_strike=Decimal("100"),
                cap_strike=Decimal("199.99"),
                won=winner == "a",
            ),
            Bracket(
                ticker=f"{event_ticker}-BAND-B",
                event_ticker=event_ticker,
                floor_strike=Decimal("200"),
                cap_strike=Decimal("299.99"),
                won=winner == "b",
            ),
            Bracket(
                ticker=f"{event_ticker}-BAND-C",
                event_ticker=event_ticker,
                floor_strike=Decimal("300"),
                cap_strike=Decimal("399.99"),
                won=winner == "c",
            ),
        )
    return SettledEvent(
        event=Event(
            event_ticker=event_ticker,
            close_ts=close,
            status=STATUS_COMPLETE,
            expiration_value=Decimal(expiration),
        ),
        brackets=brackets,
    )


def synthetic_settled_gap_ab(
    *,
    label: str,
    close: datetime,
    expiration: str,
    winner: str,
) -> SettledEvent:
    """A then jump to C (gap) — used by impulse/large-move strategies."""
    event_ticker = synth_event_ticker(label)
    return SettledEvent(
        event=Event(
            event_ticker=event_ticker,
            close_ts=close,
            status=STATUS_COMPLETE,
            expiration_value=Decimal(expiration),
        ),
        brackets=(
            Bracket(
                ticker=f"{event_ticker}-BAND-A",
                event_ticker=event_ticker,
                floor_strike=Decimal("100"),
                cap_strike=Decimal("199.99"),
                won=winner == "a",
            ),
            Bracket(
                ticker=f"{event_ticker}-BAND-C",
                event_ticker=event_ticker,
                floor_strike=Decimal("300"),
                cap_strike=Decimal("399.99"),
                won=winner == "c",
            ),
        ),
    )


def point_bar(close: datetime, minute: int, price: str) -> MinuteBar:
    return MinuteBar(
        end_ts=checkpoint_end_ts(close, minute),
        open=price,
        high=price,
        low=price,
        close=price,
    )


def ohlc_bar(
    close: datetime,
    minute: int,
    *,
    open_: str,
    high: str,
    low: str,
    close_px: str,
) -> MinuteBar:
    return MinuteBar(
        end_ts=checkpoint_end_ts(close, minute),
        open=open_,
        high=high,
        low=low,
        close=close_px,
    )


def flat_bars(close: datetime, minutes: range, price: str) -> list[MinuteBar]:
    return [point_bar(close, m, price) for m in minutes]


def hour_bars(
    close: datetime,
    *,
    price: str,
    high: str | None = None,
    low: str | None = None,
    through: int = 59,
    include_close_bar: bool = False,
) -> list[MinuteBar]:
    hi = high if high is not None else price
    lo = low if low is not None else price
    bars = [
        ohlc_bar(close, m, open_=price, high=hi, low=lo, close_px=price)
        for m in range(1, through + 1)
    ]
    if include_close_bar:
        bars.append(
            MinuteBar(
                end_ts=int(close.timestamp()),
                open=price,
                high=hi,
                low=lo,
                close=price,
            )
        )
    return bars


def _seed_event(
    connection,
    *,
    label: str,
    slot: int,
    expiration: str,
    winner: str,
    bars: Sequence[MinuteBar],
    open_high: bool = False,
    gap_ab: bool = False,
) -> datetime:
    close = synth_close(slot)
    if gap_ab:
        settled = synthetic_settled_gap_ab(
            label=label, close=close, expiration=expiration, winner=winner
        )
    else:
        settled = synthetic_settled(
            label=label,
            close=close,
            expiration=expiration,
            winner=winner,
            open_high=open_high,
        )
    store_settled_events(connection, [settled])
    store_btc_candles(connection, list(bars))
    return close


# --- per-strategy seeds (slot numbers keep candle hours unique) ---


def seed_s1_trade_yes(connection) -> None:
    close = synth_close(1)
    _seed_event(
        connection,
        label="001-TRADE",
        slot=1,
        expiration="150",
        winner="a",
        bars=flat_bars(close, range(45, 46), "150"),
    )


def seed_s1_trade_no(connection) -> None:
    seed_s1_trade_yes(connection)


def seed_s1_loss(connection) -> None:
    close = synth_close(2)
    _seed_event(
        connection,
        label="001-LOSS",
        slot=2,
        expiration="250",
        winner="b",
        bars=flat_bars(close, range(50, 51), "150"),
    )


def seed_s1_multi_event(connection) -> None:
    for i, slot in enumerate((3, 4)):
        close = synth_close(slot)
        _seed_event(
            connection,
            label=f"001-MULTI-{i}",
            slot=slot,
            expiration="150",
            winner="a",
            bars=flat_bars(close, range(55, 56), "150"),
        )


def seed_s2_trade_yes(connection) -> None:
    close = synth_close(10)
    bars = [
        ohlc_bar(close, m, open_="150", high="160", low="140", close_px="150")
        for m in range(51, 56)
    ]
    _seed_event(
        connection,
        label="002-TRADE",
        slot=10,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s2_trade_no(connection) -> None:
    seed_s2_trade_yes(connection)


def seed_s2_abstain_near_edge(connection) -> None:
    close = synth_close(11)
    bars = [
        ohlc_bar(close, m, open_="110", high="120", low="105", close_px="110")
        for m in range(51, 56)
    ]
    _seed_event(
        connection,
        label="002-NEAR",
        slot=11,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s2_abstain_unstable(connection) -> None:
    close = synth_close(12)
    bars = [
        ohlc_bar(close, m, open_="150", high="160", low="140", close_px="150")
        for m in range(51, 56)
    ]
    bars[2] = ohlc_bar(close, 53, open_="150", high="210", low="140", close_px="150")
    _seed_event(
        connection,
        label="002-UNSTABLE",
        slot=12,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s2_loss(connection) -> None:
    close = synth_close(13)
    bars = [
        ohlc_bar(close, m, open_="150", high="160", low="140", close_px="150")
        for m in range(51, 56)
    ]
    _seed_event(
        connection,
        label="002-LOSS",
        slot=13,
        expiration="250",
        winner="b",
        bars=bars,
    )


def seed_s3_trade_yes(connection) -> None:
    close = synth_close(20)
    bars = [
        point_bar(close, minute, str(200 + offset * 10))
        for offset, minute in enumerate(range(46, 56))
    ]
    _seed_event(
        connection,
        label="003-TRADE",
        slot=20,
        expiration="350",
        winner="c",
        bars=bars,
    )


def seed_s3_trade_no(connection) -> None:
    seed_s3_trade_yes(connection)


def seed_s3_loss(connection) -> None:
    close = synth_close(21)
    bars = [
        point_bar(close, minute, str(200 + offset * 10))
        for offset, minute in enumerate(range(46, 56))
    ]
    _seed_event(
        connection,
        label="003-LOSS",
        slot=21,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s3_multi_event(connection) -> None:
    for i, slot in enumerate((22, 23)):
        close = synth_close(slot)
        bars = [
            point_bar(close, minute, str(200 + offset * 10))
            for offset, minute in enumerate(range(46, 56))
        ]
        _seed_event(
            connection,
            label=f"003-MULTI-{i}",
            slot=slot,
            expiration="350",
            winner="c",
            bars=bars,
        )


def seed_s4_trade_yes(connection) -> None:
    close = synth_close(30)
    bars = [point_bar(close, 1, "100")]
    bars.extend(flat_bars(close, range(2, 56), "400"))
    _seed_event(
        connection,
        label="004-TRADE",
        slot=30,
        expiration="250",
        winner="b",
        bars=bars,
    )


def seed_s4_trade_no(connection) -> None:
    seed_s4_trade_yes(connection)


def seed_s4_abstain_small_move(connection) -> None:
    close = synth_close(31)
    _seed_event(
        connection,
        label="004-SMALL",
        slot=31,
        expiration="150",
        winner="a",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s4_loss(connection) -> None:
    close = synth_close(32)
    bars = [point_bar(close, 1, "100")]
    bars.extend(flat_bars(close, range(2, 56), "400"))
    _seed_event(
        connection,
        label="004-LOSS",
        slot=32,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s5_trade_yes(connection) -> None:
    close = synth_close(40)
    prices = [150, 151, 150, 151, 150, 151, 150, 151, 150, 151, 150]
    bars = [
        point_bar(close, minute, str(price))
        for price, minute in zip(prices, range(45, 56), strict=True)
    ]
    _seed_event(
        connection,
        label="005-TRADE",
        slot=40,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s5_trade_no(connection) -> None:
    seed_s5_trade_yes(connection)


def seed_s5_abstain_thin_buffer(connection) -> None:
    close = synth_close(41)
    prices = [150, 190, 120, 190, 120, 190, 120, 190, 120, 190, 195]
    bars = [
        point_bar(close, minute, str(price))
        for price, minute in zip(prices, range(45, 56), strict=True)
    ]
    _seed_event(
        connection,
        label="005-THIN",
        slot=41,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s5_loss(connection) -> None:
    close = synth_close(42)
    prices = [150, 151, 150, 151, 150, 151, 150, 151, 150, 151, 150]
    bars = [
        point_bar(close, minute, str(price))
        for price, minute in zip(prices, range(45, 56), strict=True)
    ]
    _seed_event(
        connection,
        label="005-LOSS",
        slot=42,
        expiration="250",
        winner="b",
        bars=bars,
    )


def seed_s6_trade_yes(connection) -> None:
    close = synth_close(50)
    bars = flat_bars(close, range(1, 53), "150")
    for minute, price in ((53, "170"), (54, "180"), (55, "190")):
        bars.append(point_bar(close, minute, price))
    _seed_event(
        connection,
        label="006-TRADE",
        slot=50,
        expiration="250",
        winner="b",
        bars=bars,
    )


def seed_s6_trade_no(connection) -> None:
    seed_s6_trade_yes(connection)


def seed_s6_abstain_no_direction(connection) -> None:
    close = synth_close(51)
    _seed_event(
        connection,
        label="006-FLAT",
        slot=51,
        expiration="150",
        winner="a",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s6_abstain_not_near_edge(connection) -> None:
    close = synth_close(52)
    bars = flat_bars(close, range(1, 53), "150")
    for minute, price in ((53, "151"), (54, "152"), (55, "153")):
        bars.append(point_bar(close, minute, price))
    _seed_event(
        connection,
        label="006-MID",
        slot=52,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s6_loss(connection) -> None:
    close = synth_close(53)
    bars = flat_bars(close, range(1, 53), "150")
    for minute, price in ((53, "170"), (54, "180"), (55, "190")):
        bars.append(point_bar(close, minute, price))
    _seed_event(
        connection,
        label="006-LOSS",
        slot=53,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s7_trade_yes(connection) -> None:
    close = synth_close(60)
    bars = flat_bars(close, range(1, 50), "150")
    bars.extend(flat_bars(close, range(50, 56), "250"))
    _seed_event(
        connection,
        label="007-TRADE",
        slot=60,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s7_trade_no(connection) -> None:
    seed_s7_trade_yes(connection)


def seed_s7_abstain_no_unique_mode(connection) -> None:
    close = synth_close(61)
    # 20×A + 20×B + 15×C → tied plurality (no unique mode)
    bars = []
    for m in range(1, 56):
        if m <= 20:
            price = "150"
        elif m <= 40:
            price = "250"
        else:
            price = "350"
        bars.append(point_bar(close, m, price))
    _seed_event(
        connection,
        label="007-TIE",
        slot=61,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s7_loss(connection) -> None:
    close = synth_close(62)
    bars = flat_bars(close, range(1, 50), "150")
    bars.extend(flat_bars(close, range(50, 56), "250"))
    _seed_event(
        connection,
        label="007-LOSS",
        slot=62,
        expiration="250",
        winner="b",
        bars=bars,
    )


def seed_s8_trade_yes(connection) -> None:
    close = synth_close(70)
    _seed_event(
        connection,
        label="008-TRADE",
        slot=70,
        expiration="150",
        winner="a",
        bars=flat_bars(close, range(50, 56), "150"),
    )


def seed_s8_trade_no(connection) -> None:
    seed_s8_trade_yes(connection)


def seed_s8_abstain_unconfirmed(connection) -> None:
    close = synth_close(71)
    bars = flat_bars(close, range(50, 51), "250")
    bars.extend(flat_bars(close, range(55, 56), "150"))
    _seed_event(
        connection,
        label="008-UNCONF",
        slot=71,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s8_loss(connection) -> None:
    close = synth_close(72)
    _seed_event(
        connection,
        label="008-LOSS",
        slot=72,
        expiration="250",
        winner="b",
        bars=flat_bars(close, range(50, 56), "150"),
    )


def seed_s9_trade_yes(connection) -> None:
    close = synth_close(80)
    bars = [
        ohlc_bar(close, m, open_="150", high="160", low="140", close_px="150")
        for m in range(51, 56)
    ]
    _seed_event(
        connection,
        label="009-TRADE",
        slot=80,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s9_trade_no(connection) -> None:
    seed_s9_trade_yes(connection)


def seed_s9_abstain_thin(connection) -> None:
    close = synth_close(81)
    bars = [
        ohlc_bar(close, m, open_="150", high="160", low="140", close_px="150")
        for m in range(51, 56)
    ]
    bars[2] = ohlc_bar(close, 53, open_="150", high="210", low="140", close_px="150")
    _seed_event(
        connection,
        label="009-THIN",
        slot=81,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s9_loss(connection) -> None:
    close = synth_close(82)
    bars = [
        ohlc_bar(close, m, open_="150", high="160", low="140", close_px="150")
        for m in range(51, 56)
    ]
    _seed_event(
        connection,
        label="009-LOSS",
        slot=82,
        expiration="250",
        winner="b",
        bars=bars,
    )


def seed_s10_trade_yes(connection) -> None:
    close = synth_close(90)
    bars = [
        ohlc_bar(close, m, open_="150", high="160", low="140", close_px="150")
        for m in range(51, 54)
    ]
    bars.extend(
        ohlc_bar(close, m, open_="150", high="210", low="140", close_px="150")
        for m in range(54, 56)
    )
    _seed_event(
        connection,
        label="010-TRADE",
        slot=90,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s10_trade_no(connection) -> None:
    seed_s10_trade_yes(connection)


def seed_s10_abstain_low_dwell(connection) -> None:
    close = synth_close(91)
    bars = [
        ohlc_bar(close, m, open_="150", high="160", low="140", close_px="150")
        for m in range(51, 53)
    ]
    bars.extend(
        ohlc_bar(close, m, open_="150", high="210", low="140", close_px="150")
        for m in range(53, 56)
    )
    _seed_event(
        connection,
        label="010-LOW",
        slot=91,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s10_loss(connection) -> None:
    close = synth_close(92)
    bars = [
        ohlc_bar(close, m, open_="150", high="160", low="140", close_px="150")
        for m in range(51, 54)
    ]
    bars.extend(
        ohlc_bar(close, m, open_="150", high="210", low="140", close_px="150")
        for m in range(54, 56)
    )
    _seed_event(
        connection,
        label="010-LOSS",
        slot=92,
        expiration="250",
        winner="b",
        bars=bars,
    )


def seed_s11_trade_yes(connection) -> None:
    close = synth_close(100)
    bars = [point_bar(close, 50, "150")]
    bars.extend(
        ohlc_bar(close, m, open_="150", high="160", low="140", close_px="150")
        for m in range(51, 56)
    )
    _seed_event(
        connection,
        label="011-TRADE",
        slot=100,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s11_trade_no(connection) -> None:
    seed_s11_trade_yes(connection)


def seed_s11_abstain_breach_up(connection) -> None:
    close = synth_close(101)
    bars = [point_bar(close, 50, "150")]
    bars.extend(
        ohlc_bar(close, m, open_="150", high="210", low="140", close_px="150")
        for m in range(51, 56)
    )
    _seed_event(
        connection,
        label="011-UP",
        slot=101,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s11_loss(connection) -> None:
    close = synth_close(102)
    bars = [point_bar(close, 50, "150")]
    bars.extend(
        ohlc_bar(close, m, open_="150", high="160", low="140", close_px="150")
        for m in range(51, 56)
    )
    _seed_event(
        connection,
        label="011-LOSS",
        slot=102,
        expiration="250",
        winner="b",
        bars=bars,
    )


def seed_s12_trade_yes(connection) -> None:
    close = synth_close(110)
    _seed_event(
        connection,
        label="012-TRADE",
        slot=110,
        expiration="150",
        winner="a",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s12_trade_no(connection) -> None:
    seed_s12_trade_yes(connection)


def seed_s12_abstain_print_risk(connection) -> None:
    close = synth_close(111)
    bars = [
        point_bar(close, m, "110" if (m // 5) % 2 == 0 else "190")
        for m in range(1, 56)
    ]
    _seed_event(
        connection,
        label="012-RISK",
        slot=111,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s12_loss(connection) -> None:
    close = synth_close(112)
    _seed_event(
        connection,
        label="012-LOSS",
        slot=112,
        expiration="250",
        winner="b",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s13_trade_yes(connection) -> None:
    # prior hour + current hour quiet
    prior = synth_close(120)
    current = synth_close(121)
    store_settled_events(
        connection,
        [
            synthetic_settled(
                label="013-PRIOR", close=prior, expiration="150", winner="a"
            ),
            synthetic_settled(
                label="013-TRADE", close=current, expiration="150", winner="a"
            ),
        ],
    )
    for close in (prior, current):
        store_btc_candles(
            connection,
            hour_bars(close, price="150", high="151", low="149", through=59),
        )


def seed_s13_trade_no(connection) -> None:
    seed_s13_trade_yes(connection)


def seed_s13_abstain_low_prob(connection) -> None:
    prior = synth_close(122)
    current = synth_close(123)
    store_settled_events(
        connection,
        [
            synthetic_settled(
                label="013-PRIOR-WILD", close=prior, expiration="150", winner="a"
            ),
            synthetic_settled(
                label="013-LOW", close=current, expiration="150", winner="a"
            ),
        ],
    )
    for close in (prior, current):
        store_btc_candles(
            connection,
            hour_bars(close, price="150", high="250", low="50", through=59),
        )


def seed_s13_loss(connection) -> None:
    prior = synth_close(124)
    current = synth_close(125)
    store_settled_events(
        connection,
        [
            synthetic_settled(
                label="013-PRIOR-L", close=prior, expiration="150", winner="a"
            ),
            synthetic_settled(
                label="013-LOSS", close=current, expiration="250", winner="b"
            ),
        ],
    )
    for close in (prior, current):
        store_btc_candles(
            connection,
            hour_bars(close, price="150", high="151", low="149", through=59),
        )


def seed_s14_trade_yes(connection) -> None:
    close = synth_close(130)
    # m1-30 wild, m31-59 calm (storm then quiet)
    bars = [
        ohlc_bar(close, m, open_="150", high="250", low="50", close_px="150")
        for m in range(1, 31)
    ]
    bars.extend(
        ohlc_bar(close, m, open_="150", high="151", low="149", close_px="150")
        for m in range(31, 60)
    )
    _seed_event(
        connection,
        label="014-TRADE",
        slot=130,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s14_trade_no(connection) -> None:
    seed_s14_trade_yes(connection)


def seed_s14_abstain_already_priced(connection) -> None:
    close = synth_close(131)
    _seed_event(
        connection,
        label="014-PRICED",
        slot=131,
        expiration="150",
        winner="a",
        bars=hour_bars(close, price="150", high="151", low="149", through=59),
    )


def seed_s14_abstain_low_quality(connection) -> None:
    close = synth_close(132)
    _seed_event(
        connection,
        label="014-WILD",
        slot=132,
        expiration="150",
        winner="a",
        bars=hour_bars(close, price="150", high="250", low="50", through=59),
    )


def seed_s14_loss(connection) -> None:
    close = synth_close(133)
    bars = [
        ohlc_bar(close, m, open_="150", high="250", low="50", close_px="150")
        for m in range(1, 31)
    ]
    bars.extend(
        ohlc_bar(close, m, open_="150", high="151", low="149", close_px="150")
        for m in range(31, 60)
    )
    _seed_event(
        connection,
        label="014-LOSS",
        slot=133,
        expiration="250",
        winner="b",
        bars=bars,
    )


def seed_s15_trade_yes(connection) -> None:
    close = synth_close(140)
    bars = [point_bar(close, 1, "100")]
    for m in range(2, 42):
        bars.append(point_bar(close, m, "100"))
    for m in range(42, 60):
        bars.append(point_bar(close, m, "350"))
    _seed_event(
        connection,
        label="015-TRADE",
        slot=140,
        expiration="350",
        winner="c",
        bars=bars,
        gap_ab=True,
    )


def seed_s15_trade_no(connection) -> None:
    seed_s15_trade_yes(connection)


def seed_s15_abstain_small_move(connection) -> None:
    close = synth_close(141)
    bars = flat_bars(close, range(1, 42), "150")
    bars.extend(flat_bars(close, range(42, 60), "160"))
    _seed_event(
        connection,
        label="015-SMALL",
        slot=141,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s15_abstain_in_transit(connection) -> None:
    close = synth_close(142)
    bars = flat_bars(close, range(1, 45), "100")
    bars.extend(flat_bars(close, range(45, 60), "350"))
    _seed_event(
        connection,
        label="015-TRANSIT",
        slot=142,
        expiration="350",
        winner="c",
        bars=bars,
        gap_ab=True,
    )


def seed_s15_loss(connection) -> None:
    close = synth_close(143)
    bars = [point_bar(close, 1, "100")]
    for m in range(2, 42):
        bars.append(point_bar(close, m, "100"))
    for m in range(42, 60):
        bars.append(point_bar(close, m, "350"))
    _seed_event(
        connection,
        label="015-LOSS",
        slot=143,
        expiration="150",
        winner="a",
        bars=bars,
        gap_ab=True,
    )


def seed_s16_trade_yes(connection) -> None:
    close = synth_close(150)
    _seed_event(
        connection,
        label="016-TRADE",
        slot=150,
        expiration="150",
        winner="a",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s16_trade_no(connection) -> None:
    seed_s16_trade_yes(connection)


def seed_s16_abstain_unconfirmed(connection) -> None:
    close = synth_close(151)
    bars = [point_bar(close, 1, "150"), point_bar(close, 50, "250")]
    bars.append(point_bar(close, 55, "150"))
    _seed_event(
        connection,
        label="016-UNCONF",
        slot=151,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s16_abstain_large_move(connection) -> None:
    close = synth_close(152)
    bars = [
        point_bar(close, 1, "100"),
        point_bar(close, 50, "350"),
        point_bar(close, 55, "350"),
    ]
    _seed_event(
        connection,
        label="016-LARGE",
        slot=152,
        expiration="350",
        winner="c",
        bars=bars,
        gap_ab=True,
    )


def seed_s16_loss(connection) -> None:
    close = synth_close(153)
    _seed_event(
        connection,
        label="016-LOSS",
        slot=153,
        expiration="250",
        winner="b",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s17_trade_yes(connection) -> None:
    close = synth_close(160)
    _seed_event(
        connection,
        label="017-TRADE",
        slot=160,
        expiration="150",
        winner="a",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s17_trade_no(connection) -> None:
    seed_s17_trade_yes(connection)


def seed_s17_abstain_mode_not_current(connection) -> None:
    close = synth_close(161)
    bars = flat_bars(close, range(1, 40), "250")
    bars.extend(flat_bars(close, range(40, 56), "150"))
    _seed_event(
        connection,
        label="017-MODE",
        slot=161,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s17_abstain_low_occupancy(connection) -> None:
    close = synth_close(162)
    bars = [
        point_bar(close, m, "250" if m % 2 else "150") for m in range(1, 56)
    ]
    _seed_event(
        connection,
        label="017-OCC",
        slot=162,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s17_loss(connection) -> None:
    close = synth_close(163)
    _seed_event(
        connection,
        label="017-LOSS",
        slot=163,
        expiration="250",
        winner="b",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s18_trade_yes(connection) -> None:
    close = synth_close(170)
    bars = flat_bars(close, range(1, 51), "150")
    bars.extend(flat_bars(close, range(51, 56), "125"))
    _seed_event(
        connection,
        label="018-TRADE",
        slot=170,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s18_trade_no(connection) -> None:
    seed_s18_trade_yes(connection)


def seed_s18_abstain_unconfirmed(connection) -> None:
    close = synth_close(171)
    bars = flat_bars(close, range(1, 51), "250")
    bars.extend(flat_bars(close, range(51, 56), "150"))
    _seed_event(
        connection,
        label="018-UNCONF",
        slot=171,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s18_abstain_not_edgeward(connection) -> None:
    close = synth_close(172)
    bars = flat_bars(close, range(1, 51), "125")
    bars.extend(flat_bars(close, range(51, 56), "150"))
    _seed_event(
        connection,
        label="018-CENTER",
        slot=172,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s18_loss(connection) -> None:
    close = synth_close(173)
    bars = flat_bars(close, range(1, 51), "150")
    bars.extend(flat_bars(close, range(51, 56), "125"))
    _seed_event(
        connection,
        label="018-LOSS",
        slot=173,
        expiration="250",
        winner="b",
        bars=bars,
    )


def seed_s19_trade_yes(connection) -> None:
    close = synth_close(180)
    bars = flat_bars(close, range(1, 51), "150")
    bars.extend(flat_bars(close, range(51, 53), "250"))
    bars.extend(flat_bars(close, range(53, 56), "150"))
    _seed_event(
        connection,
        label="019-TRADE",
        slot=180,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s19_trade_no(connection) -> None:
    seed_s19_trade_yes(connection)


def seed_s19_abstain_no_excursion(connection) -> None:
    close = synth_close(181)
    _seed_event(
        connection,
        label="019-FLAT",
        slot=181,
        expiration="150",
        winner="a",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s19_abstain_unconfirmed(connection) -> None:
    close = synth_close(182)
    bars = flat_bars(close, range(1, 51), "250")
    bars.extend(flat_bars(close, range(51, 56), "150"))
    _seed_event(
        connection,
        label="019-UNCONF",
        slot=182,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s19_loss(connection) -> None:
    close = synth_close(183)
    bars = flat_bars(close, range(1, 51), "150")
    bars.extend(flat_bars(close, range(51, 53), "250"))
    bars.extend(flat_bars(close, range(53, 56), "150"))
    _seed_event(
        connection,
        label="019-LOSS",
        slot=183,
        expiration="250",
        winner="b",
        bars=bars,
    )


def seed_s20_abstain_insufficient(connection) -> None:
    close = synth_close(190)
    _seed_event(
        connection,
        label="020-FIRST",
        slot=190,
        expiration="150",
        winner="a",
        bars=hour_bars(close, price="150", include_close_bar=True),
    )


def seed_s20_trade_yes(connection) -> None:
    for i, slot in enumerate((191, 192, 193)):
        close = synth_close(slot)
        _seed_event(
            connection,
            label=f"020-LIB-{i}",
            slot=slot,
            expiration="150",
            winner="a",
            bars=hour_bars(close, price="150", include_close_bar=True),
        )


def seed_s20_trade_no(connection) -> None:
    seed_s20_trade_yes(connection)


def seed_s20_loss(connection) -> None:
    for i, slot in enumerate((194, 195, 196)):
        winner = "a" if i < 2 else "b"
        expiration = "150" if i < 2 else "250"
        close = synth_close(slot)
        # Keep path in A so vote stays on A; last event settles B → loss.
        price = "150"
        _seed_event(
            connection,
            label=f"020-LOSS-{i}",
            slot=slot,
            expiration=expiration,
            winner=winner,
            bars=hour_bars(close, price=price, include_close_bar=True),
        )


def seed_s20_multi_event(connection) -> None:
    seed_s20_trade_yes(connection)


def seed_s21_trade_yes(connection) -> None:
    close = synth_close(200)
    _seed_event(
        connection,
        label="021-TRADE",
        slot=200,
        expiration="150",
        winner="a",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s21_trade_no(connection) -> None:
    seed_s21_trade_yes(connection)


def seed_s21_abstain_weak_lock(connection) -> None:
    close = synth_close(201)
    bars = flat_bars(close, range(1, 50), "150")
    bars.extend(flat_bars(close, range(50, 53), "180"))
    bars.extend(flat_bars(close, range(53, 56), "105"))
    _seed_event(
        connection,
        label="021-WEAK",
        slot=201,
        expiration="105",
        winner="a",
        bars=bars,
    )


def seed_s21_loss(connection) -> None:
    close = synth_close(202)
    _seed_event(
        connection,
        label="021-LOSS",
        slot=202,
        expiration="250",
        winner="b",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s22_trade_yes(connection) -> None:
    close = synth_close(210)
    _seed_event(
        connection,
        label="022-TRADE",
        slot=210,
        expiration="150",
        winner="a",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s22_trade_no(connection) -> None:
    seed_s22_trade_yes(connection)


def seed_s22_abstain_asymmetric_weak(connection) -> None:
    close = synth_close(211)
    bars = flat_bars(close, range(1, 50), "150")
    bars.extend(flat_bars(close, range(50, 53), "180"))
    bars.extend(flat_bars(close, range(53, 56), "105"))
    _seed_event(
        connection,
        label="022-WEAK",
        slot=211,
        expiration="105",
        winner="a",
        bars=bars,
    )


def seed_s22_loss(connection) -> None:
    close = synth_close(212)
    _seed_event(
        connection,
        label="022-LOSS",
        slot=212,
        expiration="250",
        winner="b",
        bars=flat_bars(close, range(1, 56), "150"),
    )


def seed_s23_trade_yes(connection) -> None:
    close = synth_close(220)
    bars = flat_bars(close, range(1, 31), "160")
    bars.extend(flat_bars(close, range(31, 46), "145"))
    _seed_event(
        connection,
        label="023-TRADE",
        slot=220,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s23_trade_no(connection) -> None:
    seed_s23_trade_yes(connection)


def seed_s23_abstain_young_age(connection) -> None:
    close = synth_close(221)
    bars = flat_bars(close, range(1, 36), "250")
    bars.extend(flat_bars(close, range(36, 46), "145"))
    _seed_event(
        connection,
        label="023-YOUNG",
        slot=221,
        expiration="150",
        winner="a",
        bars=bars,
    )


def seed_s23_abstain_finished(connection) -> None:
    close = synth_close(222)
    _seed_event(
        connection,
        label="023-FLAT",
        slot=222,
        expiration="150",
        winner="a",
        bars=flat_bars(close, range(1, 46), "150"),
    )


def seed_s23_loss(connection) -> None:
    close = synth_close(223)
    bars = flat_bars(close, range(1, 31), "160")
    bars.extend(flat_bars(close, range(31, 46), "145"))
    _seed_event(
        connection,
        label="023-LOSS",
        slot=223,
        expiration="250",
        winner="b",
        bars=bars,
    )


def _case(
    case_id: str,
    strategy: str,
    seed: SeedFn,
    *,
    minutes: tuple[int, ...] = (55,),
    side: Side = "yes",
    eval_kwargs: Mapping[str, Any] | None = None,
) -> GoldenCase:
    return GoldenCase(
        case_id=case_id,
        strategy=strategy,
        minutes=minutes,
        side=side,
        seed=seed,
        eval_kwargs=eval_kwargs,
    )


GOLDEN_CASES: tuple[GoldenCase, ...] = (
    # s1
    _case("s1/trade_yes", "s1", seed_s1_trade_yes, minutes=(45,)),
    _case("s1/trade_no", "s1", seed_s1_trade_no, minutes=(45,), side="no"),
    _case("s1/loss", "s1", seed_s1_loss, minutes=(50,)),
    _case("s1/multi_event", "s1", seed_s1_multi_event, minutes=(55,)),
    # s2
    _case("s2/trade_yes", "s2", seed_s2_trade_yes),
    _case("s2/trade_no", "s2", seed_s2_trade_no, side="no"),
    _case("s2/abstain_near_edge", "s2", seed_s2_abstain_near_edge),
    _case("s2/abstain_unstable", "s2", seed_s2_abstain_unstable),
    _case("s2/loss", "s2", seed_s2_loss),
    # s3
    _case("s3/trade_yes", "s3", seed_s3_trade_yes),
    _case("s3/trade_no", "s3", seed_s3_trade_no, side="no"),
    _case("s3/loss", "s3", seed_s3_loss),
    _case("s3/multi_event", "s3", seed_s3_multi_event),
    # s4
    _case("s4/trade_yes", "s4", seed_s4_trade_yes),
    _case("s4/trade_no", "s4", seed_s4_trade_no, side="no"),
    _case("s4/abstain_small_move", "s4", seed_s4_abstain_small_move),
    _case("s4/loss", "s4", seed_s4_loss),
    # s5
    _case("s5/trade_yes", "s5", seed_s5_trade_yes),
    _case("s5/trade_no", "s5", seed_s5_trade_no, side="no"),
    _case("s5/abstain_thin_buffer", "s5", seed_s5_abstain_thin_buffer),
    _case("s5/loss", "s5", seed_s5_loss),
    # s6
    _case("s6/trade_yes", "s6", seed_s6_trade_yes),
    _case("s6/trade_no", "s6", seed_s6_trade_no, side="no"),
    _case("s6/abstain_no_direction", "s6", seed_s6_abstain_no_direction),
    _case("s6/abstain_not_near_edge", "s6", seed_s6_abstain_not_near_edge),
    _case("s6/loss", "s6", seed_s6_loss),
    # s7
    _case("s7/trade_yes", "s7", seed_s7_trade_yes),
    _case("s7/trade_no", "s7", seed_s7_trade_no, side="no"),
    _case("s7/abstain_no_unique_mode", "s7", seed_s7_abstain_no_unique_mode),
    _case("s7/loss", "s7", seed_s7_loss),
    # s8
    _case("s8/trade_yes", "s8", seed_s8_trade_yes),
    _case("s8/trade_no", "s8", seed_s8_trade_no, side="no"),
    _case("s8/abstain_unconfirmed", "s8", seed_s8_abstain_unconfirmed),
    _case("s8/loss", "s8", seed_s8_loss),
    # s9
    _case("s9/trade_yes", "s9", seed_s9_trade_yes),
    _case("s9/trade_no", "s9", seed_s9_trade_no, side="no"),
    _case("s9/abstain_thin", "s9", seed_s9_abstain_thin),
    _case("s9/loss", "s9", seed_s9_loss),
    # s10
    _case("s10/trade_yes", "s10", seed_s10_trade_yes),
    _case("s10/trade_no", "s10", seed_s10_trade_no, side="no"),
    _case("s10/abstain_low_dwell", "s10", seed_s10_abstain_low_dwell),
    _case("s10/loss", "s10", seed_s10_loss),
    # s11
    _case("s11/trade_yes", "s11", seed_s11_trade_yes),
    _case("s11/trade_no", "s11", seed_s11_trade_no, side="no"),
    _case("s11/abstain_breach_up", "s11", seed_s11_abstain_breach_up),
    _case("s11/loss", "s11", seed_s11_loss),
    # s12
    _case("s12/trade_yes", "s12", seed_s12_trade_yes),
    _case("s12/trade_no", "s12", seed_s12_trade_no, side="no"),
    _case("s12/abstain_print_risk", "s12", seed_s12_abstain_print_risk),
    _case("s12/loss", "s12", seed_s12_loss),
    # s13
    _case("s13/trade_yes", "s13", seed_s13_trade_yes),
    _case("s13/trade_no", "s13", seed_s13_trade_no, side="no"),
    _case("s13/abstain_low_prob", "s13", seed_s13_abstain_low_prob),
    _case("s13/loss", "s13", seed_s13_loss),
    # s14
    _case("s14/trade_yes", "s14", seed_s14_trade_yes, minutes=(45,)),
    _case("s14/trade_no", "s14", seed_s14_trade_no, minutes=(45,), side="no"),
    _case(
        "s14/abstain_already_priced",
        "s14",
        seed_s14_abstain_already_priced,
        minutes=(45,),
    ),
    _case(
        "s14/abstain_low_quality",
        "s14",
        seed_s14_abstain_low_quality,
        minutes=(45,),
    ),
    _case("s14/loss", "s14", seed_s14_loss, minutes=(45,)),
    # s15
    _case("s15/trade_yes", "s15", seed_s15_trade_yes, minutes=(45,)),
    _case("s15/trade_no", "s15", seed_s15_trade_no, minutes=(45,), side="no"),
    _case(
        "s15/abstain_small_move",
        "s15",
        seed_s15_abstain_small_move,
        minutes=(45,),
    ),
    _case(
        "s15/abstain_in_transit",
        "s15",
        seed_s15_abstain_in_transit,
        minutes=(45,),
    ),
    _case("s15/loss", "s15", seed_s15_loss, minutes=(45,)),
    # s16
    _case("s16/trade_yes", "s16", seed_s16_trade_yes),
    _case("s16/trade_no", "s16", seed_s16_trade_no, side="no"),
    _case("s16/abstain_unconfirmed", "s16", seed_s16_abstain_unconfirmed),
    _case("s16/abstain_large_move", "s16", seed_s16_abstain_large_move),
    _case("s16/loss", "s16", seed_s16_loss),
    # s17
    _case("s17/trade_yes", "s17", seed_s17_trade_yes),
    _case("s17/trade_no", "s17", seed_s17_trade_no, side="no"),
    _case(
        "s17/abstain_mode_not_current",
        "s17",
        seed_s17_abstain_mode_not_current,
    ),
    _case(
        "s17/abstain_low_occupancy",
        "s17",
        seed_s17_abstain_low_occupancy,
    ),
    _case("s17/loss", "s17", seed_s17_loss),
    # s18
    _case("s18/trade_yes", "s18", seed_s18_trade_yes),
    _case("s18/trade_no", "s18", seed_s18_trade_no, side="no"),
    _case("s18/abstain_unconfirmed", "s18", seed_s18_abstain_unconfirmed),
    _case("s18/abstain_not_edgeward", "s18", seed_s18_abstain_not_edgeward),
    _case("s18/loss", "s18", seed_s18_loss),
    # s19
    _case("s19/trade_yes", "s19", seed_s19_trade_yes),
    _case("s19/trade_no", "s19", seed_s19_trade_no, side="no"),
    _case("s19/abstain_no_excursion", "s19", seed_s19_abstain_no_excursion),
    _case("s19/abstain_unconfirmed", "s19", seed_s19_abstain_unconfirmed),
    _case("s19/loss", "s19", seed_s19_loss),
    # s20
    _case(
        "s20/abstain_insufficient",
        "s20",
        seed_s20_abstain_insufficient,
    ),
    _case("s20/trade_yes", "s20", seed_s20_trade_yes),
    _case("s20/trade_no", "s20", seed_s20_trade_no, side="no"),
    _case("s20/loss", "s20", seed_s20_loss),
    _case("s20/multi_event", "s20", seed_s20_multi_event),
    # s21
    _case("s21/trade_yes", "s21", seed_s21_trade_yes),
    _case("s21/trade_no", "s21", seed_s21_trade_no, side="no"),
    _case("s21/abstain_weak_lock", "s21", seed_s21_abstain_weak_lock),
    _case("s21/loss", "s21", seed_s21_loss),
    # s22
    _case("s22/trade_yes", "s22", seed_s22_trade_yes),
    _case("s22/trade_no", "s22", seed_s22_trade_no, side="no"),
    _case(
        "s22/abstain_asymmetric_weak",
        "s22",
        seed_s22_abstain_asymmetric_weak,
    ),
    _case("s22/loss", "s22", seed_s22_loss),
    # s23
    _case("s23/trade_yes", "s23", seed_s23_trade_yes, minutes=(45,)),
    _case("s23/trade_no", "s23", seed_s23_trade_no, minutes=(45,), side="no"),
    _case(
        "s23/abstain_young_age",
        "s23",
        seed_s23_abstain_young_age,
        minutes=(45,),
    ),
    _case(
        "s23/abstain_finished",
        "s23",
        seed_s23_abstain_finished,
        minutes=(45,),
    ),
    _case("s23/loss", "s23", seed_s23_loss, minutes=(45,)),
)

GOLDEN_CASE_BY_ID = {case.case_id: case for case in GOLDEN_CASES}
