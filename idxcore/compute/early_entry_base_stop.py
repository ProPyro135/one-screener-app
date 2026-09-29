"""Early Entry & Hard TP with its base-low first stop (strategy I).

``early_entry`` (PINESCRIPT F) as the Pine's comment intends it: the first
stop sits at the base low (the lowest low of the 20 bars before the signal),
then trails the highest high since the buy minus 3 ATR, only rising. In the
Pine itself that first stop is wiped on the signal bar, which is what F
reproduces. Everything else is F's.

A *monitoring* screener with no validated edge.
"""

from __future__ import annotations

import pandas as pd

from idxcore.compute import early_entry as f
from idxcore.compute.early_entry import MIN_BARS, _all_bars, _ticker_meta, category_of, prepare  # noqa: F401


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    return f.run_state_machine(df, base_low_stop=True)


if __name__ == "__main__":
    # Signal at bar 1 with a base low of 97; fill at bar 2 (100). F's stop
    # starts at 102 - 2x3 = 96, I's at max(97, 96) = 97. Bar 4 dips to 96.5:
    # I is stopped at 97, F rides on.
    n = 8
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": [100.0] * n, "high": [102.0] * n, "low": [98.0] * n, "close": [100.0] * n,
        "atr": [2.0] * n, "signal": [False] * n, "range_low": [97.0] * n,
    })
    frame.loc[1, "signal"] = True
    frame.loc[4, "low"] = 96.5                  # under I's 97, over F's 96
    _, t_f, _ = f.run_state_machine(frame)
    _, t_i, _ = run_state_machine(frame)
    assert t_f[0]["exit_code"] == "OPEN", t_f
    assert t_i[0]["exit_code"] == "CL Stop loss" and t_i[0]["exit_price"] == 97.0, t_i
    print("early_entry_base_stop self-check ok")
