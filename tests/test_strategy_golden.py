# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Golden same-input → same-decision locks for every research strategy."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from bob.db import connect, initialize_schema
from bob.research.runner import evaluate_strategy
from strategy_fixtures import GOLDEN_CASE_BY_ID, GOLDEN_CASES, GoldenCase

GOLDEN_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "research" / "golden_decisions.json"
)


def _minute_stats_dict(stats: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "minute": stats.minute,
        "eligible": stats.eligible,
        "wins": stats.wins,
        "losses": stats.losses,
        "exclusions": dict(stats.exclusions),
    }
    abstentions = getattr(stats, "abstentions", None)
    if abstentions is not None:
        out["abstentions"] = dict(abstentions)
    return out


def canonicalize_report(report: Any, case: GoldenCase) -> dict[str, Any]:
    return {
        "strategy": report.strategy,
        "side": report.side,
        "minutes": [_minute_stats_dict(m) for m in report.minutes],
        "trades": [
            {
                "event_ticker": t.event_ticker,
                "market_ticker": t.market_ticker,
                "minute": t.minute,
                "end_ts": t.end_ts,
                "side": t.side,
                "won": t.won,
            }
            for t in report.trades
        ],
    }


def run_case(case: GoldenCase) -> dict[str, Any]:
    connection = connect(":memory:")
    initialize_schema(connection)
    try:
        case.seed(connection)
        kwargs = dict(case.eval_kwargs or {})
        report = evaluate_strategy(
            connection,
            case.strategy,
            minutes=case.minutes,
            side=case.side,
            **kwargs,
        )
        return canonicalize_report(report, case)
    finally:
        connection.close()


def generate_golden_file(path: Path = GOLDEN_PATH) -> dict[str, Any]:
    payload = {case.case_id: run_case(case) for case in GOLDEN_CASES}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


@pytest.fixture(scope="module")
def golden_expected() -> dict[str, Any]:
    if os.environ.get("BOB_REGENERATE_GOLDEN") == "1":
        return generate_golden_file()
    if not GOLDEN_PATH.is_file():
        pytest.fail(
            f"missing {GOLDEN_PATH}; run with BOB_REGENERATE_GOLDEN=1 once to create it"
        )
    return json.loads(GOLDEN_PATH.read_text())


@pytest.mark.parametrize("case_id", [c.case_id for c in GOLDEN_CASES])
def test_golden_decision(case_id: str, golden_expected: dict[str, Any]) -> None:
    case = GOLDEN_CASE_BY_ID[case_id]
    assert case_id in golden_expected, f"golden JSON missing case {case_id}"
    actual = run_case(case)
    expected = golden_expected[case_id]
    assert actual["strategy"] == expected["strategy"] == case.strategy
    assert actual["side"] == expected["side"] == case.side
    assert actual["minutes"] == expected["minutes"]
    assert actual["trades"] == expected["trades"]
    for trade in actual["trades"]:
        assert trade["market_ticker"], "selected market must be frozen"
        assert trade["event_ticker"].startswith("SYNTH-EVENT-")
        assert "-BAND-" in trade["market_ticker"]


def test_golden_case_ids_cover_all_strategies() -> None:
    covered = {c.strategy for c in GOLDEN_CASES}
    assert covered == {f"s{i}" for i in range(1, 24)}


def test_golden_file_has_no_float_literals(golden_expected: dict[str, Any]) -> None:
    raw = GOLDEN_PATH.read_text()
    # Decimals must be strings if present; our freeze uses ints/bools/strings only.
    def _walk(node: Any) -> None:
        if isinstance(node, float):
            raise AssertionError(f"float in golden payload: {node!r}")
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)

    _walk(golden_expected)
    assert "KXBTC" not in raw
    assert "SYNTH-EVENT" in raw
