# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from decimal import Decimal

from bob.research.common import ols_slope, sample_stdev


def test_ols_slope_and_stdev() -> None:
    values = [Decimal(str(index)) for index in range(10)]
    assert ols_slope(values) == Decimal("1.0")
    assert sample_stdev([Decimal("1"), Decimal("1"), Decimal("1")]) == Decimal("0.0")
