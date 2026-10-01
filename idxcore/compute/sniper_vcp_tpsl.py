"""Sniper VCP with a fixed TP and SL (strategy K, EOD).

PINESCRIPT I's entry (``sniper_vcp``) with the exit that measured the best win
rate while staying profitable: a fixed take-profit and stop-loss from the
actual buy price. Picked on 2016-2022 entries, then checked on 2023-2026
(not used to pick): TP +7% / SL -10% won 65% at +1.0% net per trade, on 98
trades.

    TP   buy price + 7%, a resting limit
    SL   buy price - 10%, a resting stop

Both work from the bar after the fill, filled at their level or at the open
on a gap through; a bar that reaches both counts as the stop. The Pine for
this hypothesis is ``K_Sniper_VCP_TP_SL.pine``.

A *monitoring* screener with no validated edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idxcore.compute.sniper_vcp import MIN_BARS, _all_bars, _ticker_meta, category_of, prepare  # noqa: F401

TP_PCT = 7.0
SL_PCT = 10.0


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Signals at the close, fills at the next open, TP and SL resting."""
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    signal = df["signal"].to_numpy(bool)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n
    in_pos = pending_buy = False
    entry_price = np.nan
    entry_date = None
    entry_idx = -1

    def close_trade(i: int, code: str, price: float) -> None:
        codes[i] = code
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": "BUY Sniper", "exit_date": dates[i],
            "exit_price": price, "exit_code": code, "bars_held": i - entry_idx,
            "gross_return_pct": (price / entry_price - 1.0) * 100.0, "resolved": True,
        })

    for i in range(n):
        if pending_buy:
            pending_buy, in_pos = False, True
            entry_price, entry_date, entry_idx = o[i], dates[i], i
            codes[i] = "BUY Sniper"
        elif in_pos:
            tp, sl = entry_price * (1 + TP_PCT / 100), entry_price * (1 - SL_PCT / 100)
            if lo[i] <= sl:
                in_pos = False
                close_trade(i, f"CL Stop {SL_PCT:g}%", min(o[i], sl))
            elif h[i] >= tp:
                in_pos = False
                close_trade(i, f"TP Target {TP_PCT:g}%", max(o[i], tp))
        if not in_pos and not pending_buy and signal[i]:
            pending_buy = True
        position_arr[i] = int(in_pos)

    if in_pos:
        last = n - 1
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": "BUY Sniper", "exit_date": dates[last],
            "exit_price": c[last], "exit_code": "OPEN", "bars_held": last - entry_idx,
            "gross_return_pct": (c[last] / entry_price - 1.0) * 100.0, "resolved": False,
        })

    setup_arr = [False] * n
    if n and pending_buy:
        setup_arr[-1] = True
    return codes, trades, {"position": position_arr, "setup": setup_arr}


if __name__ == "__main__":
    n = 12
    f = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n, "close": [100.0] * n,
        "signal": [False] * n,
    })
    f.loc[[1, 5], "signal"] = True
    f.loc[2, "high"] = 120.0          # fill bar: no exit yet
    f.loc[3, "high"] = 108.0          # TP 107
    f.loc[7, ["low", "high"]] = [89.0, 110.0]   # both on one bar -> SL 90
    _, t, lines = run_state_machine(f)
    got = [(x["exit_code"], round(x["exit_price"], 6)) for x in t]
    assert got == [("TP Target 7%", 107.0), ("CL Stop 10%", 90.0)], got
    assert not any(lines["setup"])
    print("sniper_vcp_tpsl self-check ok")
