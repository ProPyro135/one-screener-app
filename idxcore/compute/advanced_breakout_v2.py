"""Advanced Breakout v2 strategy (strategy J, EOD).

A port of the Pine Script "Market Structure: Advanced Breakout v2" at its
defaults. Entries are PINESCRIPT E's (``advanced_breakout``); v2 adds exits:

    First stop     the highest of close - 4 ATR, the base low, and close - 7%
    Trailing stop  the highest high since the buy minus 4 ATR, only rising
    Breakeven      once the highest high is 8% over the buy, the stop is at
                   least the buy price + 1%
    Failed breakout  within the first 10 bars, a close more than 2% under
                   the breakout level (the base high)
    Time stop      from bar 15 on, if the highest high never reached +5%

All exits are decided at the close and sold at the next open, like the BUY.
The Pine's IHSG filter is off by default and so is left out. A sale over the
buy price counts as TP, otherwise as CL.

A *monitoring* screener with no validated edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idxcore.compute import accumulation_breakout as ab
from idxcore.compute.advanced_breakout import E
from idxcore.compute.accumulation_breakout import _all_bars, _ticker_meta, category_of  # noqa: F401

MIN_BARS = E.ema_len + 1
ATR_MULT = 4.0
MAX_RISK_PCT = 7.0
FAIL_BARS, FAIL_BUFFER_PCT = 10, 2.0
BE_TRIGGER_PCT, BE_LOCK_PCT = 8.0, 1.0
TIME_BARS, TIME_MIN_GAIN_PCT = 15, 5.0


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    df = ab.prepare(history, E)
    high = pd.to_numeric(df["high"], errors="coerce")
    df["range_high"] = high.shift(1).rolling(E.accum_len, min_periods=E.accum_len).max()
    return df


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Signals at the close, fills at the next open."""
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    atr = df["atr"].to_numpy(float)
    signal = df["signal"].to_numpy(bool)
    rng_lo, rng_hi = df["range_low"].to_numpy(float), df["range_high"].to_numpy(float)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n

    in_pos = pending_buy = False
    pending_sell = ""
    entry_price = stop = highest = level = np.nan
    entry_date = None
    entry_idx = -1

    for i in range(n):
        # 1. Orders from the previous close fill at this open.
        if pending_sell:
            code = ("TP " if o[i] > entry_price else "CL ") + pending_sell
            codes[i] = code
            trades.append({
                "entry_date": entry_date, "entry_price": entry_price,
                "entry_code": "BUY Breakout", "exit_date": dates[i],
                "exit_price": o[i], "exit_code": code, "bars_held": i - entry_idx,
                "gross_return_pct": (o[i] / entry_price - 1.0) * 100.0, "resolved": True,
            })
            in_pos, pending_sell = False, ""
        if pending_buy:
            pending_buy, in_pos = False, True
            entry_price, entry_date, entry_idx = o[i], dates[i], i
            codes[i] = "BUY Breakout" if not codes[i] else f"{codes[i]} + BUY Breakout"
        # 2. At the close.
        if in_pos and not pending_sell:
            bars_in = i - entry_idx
            highest = h[i] if np.isnan(highest) else max(highest, h[i])
            stop = max(stop, highest - atr[i] * ATR_MULT)
            if highest >= entry_price * (1 + BE_TRIGGER_PCT / 100):
                stop = max(stop, entry_price * (1 + BE_LOCK_PCT / 100))
            if c[i] < stop:
                pending_sell = "Stop Hit"
            elif bars_in <= FAIL_BARS and c[i] < level * (1 - FAIL_BUFFER_PCT / 100):
                pending_sell = "Failed Breakout"
            elif bars_in >= TIME_BARS and highest < entry_price * (1 + TIME_MIN_GAIN_PCT / 100):
                pending_sell = "Time Stop"
        elif not in_pos and signal[i]:
            pending_buy, highest, level = True, np.nan, rng_hi[i]
            stop = max(c[i] - atr[i] * ATR_MULT, rng_lo[i], c[i] * (1 - MAX_RISK_PCT / 100))
        position_arr[i] = int(in_pos)

    if in_pos:
        last = n - 1
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": "BUY Breakout", "exit_date": dates[last],
            "exit_price": c[last], "exit_code": "OPEN", "bars_held": last - entry_idx,
            "gross_return_pct": (c[last] / entry_price - 1.0) * 100.0, "resolved": False,
        })

    # A buy signalled at the last close, to be filled at tomorrow's open.
    setup_arr = [False] * n
    if n and pending_buy:
        setup_arr[-1] = True
    return codes, trades, {"position": position_arr, "setup": setup_arr}


if __name__ == "__main__":
    n = 30

    def frame():
        f = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n, "close": [100.0] * n,
            "atr": [1.0] * n, "signal": [False] * n, "range_low": [90.0] * n, "range_high": [99.0] * n,
        })
        f.loc[1, "signal"] = True
        return f

    # Failed breakout: level 103, the fill bar closes at 100 (< 100.94, over
    # the 97 stop) -> sold at bar 3's open.
    f = frame()
    f["range_high"] = 103.0
    _, t, _ = run_state_machine(f)
    assert t[0]["exit_code"] == "CL Failed Breakout" and pd.Timestamp(t[0]["exit_date"]) == f.loc[3, "date"], t[0]
    # First stop capped at 7%: close 100 -> max(96, 90, 93) = 96; close 95.5 -> stop hit.
    f = frame()
    f.loc[3, "close"] = 95.5
    _, t, _ = run_state_machine(f)
    assert t[0]["exit_code"] == "CL Stop Hit", t[0]
    # Time stop: flat for 15 bars after the bar-2 fill -> sold at bar 18's open.
    _, t, _ = run_state_machine(frame())
    assert t[0]["exit_code"] == "CL Time Stop" and pd.Timestamp(t[0]["exit_date"]) == frame().loc[18, "date"], t[0]
    # A run to 110 lifts the stop to 106 (trail; breakeven would give 101);
    # a close at 100.5 sells at the next open, 101.5, over the buy -> TP.
    f = frame()
    f.loc[3, "high"], f.loc[3, "close"] = 110.0, 108.0
    f.loc[4, "close"] = 100.5
    f.loc[5, "open"] = 101.5
    _, t, _ = run_state_machine(f)
    assert t[0]["exit_code"] == "TP Stop Hit" and t[0]["exit_price"] == 101.5, t[0]
    print("advanced_breakout_v2 self-check ok")
