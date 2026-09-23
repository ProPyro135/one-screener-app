"""Pullback Sniper strategy (buy a green reclaim of MA5 under the Bollinger mid).

A port of the Pine Script "Pullback Sniper + Cloud & Lines (100% Cuan TP)", in
the same shape as ``market_structure`` so ``trade_log`` serves it unchanged.

    BUY     a green bar closing above the previous close and above SMA5, still
            under the Bollinger mid (SMA20), with the mid at or above its value
            10 bars ago
    TP      close above the entry and either (a) above the mid but back under
            SMA5, or (b) the high tags the upper band on a red bar
    CL      the bar's low touches the stop: the 7-bar low at entry, capped at
            -10% from the entry

Bollinger (population stdev, as Pine), SMA5 and SMA20 are read from the stored
indicators rather than recomputed (invariant 1); ``bb_basis`` is ``ma20``.
Fills are at the signal bar's close, like the other strategies. The Pine's
strategy tester fills its CL at the stop price instead, so a CL here can sit
below the -10% cap on a bar that gaps through it.

A *monitoring* screener with no validated edge; the chart lines and cloud are
visual only and are not ported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idxcore.compute.bottom_fishing import _all_bars, _ticker_meta  # noqa: F401

MAX_CL_PCT = 10.0
CL_LOW_BARS = 7
MID_SLOPE_BARS = 10

MIN_BARS = 20 + MID_SLOPE_BARS  # the mid needs 20 bars, then 10 more for its slope

CATEGORY = {"BUY": "BUY", "TP": "TAKE PROFIT", "CL": "CUT LOSS"}


def category_of(code: str) -> str:
    return CATEGORY.get(code.split(" ", 1)[0], "OTHER")


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    """Add the 7-bar low and the mid's value 10 bars back."""
    df = history.sort_values("date").reset_index(drop=True).copy()
    low = pd.to_numeric(df["low"], errors="coerce")
    df["low7"] = low.rolling(CL_LOW_BARS, min_periods=CL_LOW_BARS).min()
    df["mid_prev"] = pd.to_numeric(df["ma20"], errors="coerce").shift(MID_SLOPE_BARS)
    return df


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Exit first, then entry, in the Pine's own order."""
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    sma5 = pd.to_numeric(df["ma5"], errors="coerce").to_numpy(float)
    mid = pd.to_numeric(df["ma20"], errors="coerce").to_numpy(float)
    upper = pd.to_numeric(df["bb_upper"], errors="coerce").to_numpy(float)
    mid_prev = df["mid_prev"].to_numpy(float)
    low7 = df["low7"].to_numpy(float)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n

    in_trade = False
    entry_price = np.nan
    entry_date = None
    entry_idx = -1
    cl_price = np.nan

    for i in range(n):
        above_entry = c[i] > entry_price  # False while entry_price is NaN, as in Pine
        tp1 = c[i] > mid[i] and c[i] < sma5[i] and above_entry
        tp2 = h[i] >= upper[i] and c[i] < o[i] and above_entry

        if in_trade:
            exit_code = ""
            if lo[i] <= cl_price:
                exit_code = "CL"
            elif tp1 or tp2:
                exit_code = "TP"
            if exit_code:
                in_trade = False
                codes[i] = exit_code
                trades.append({
                    "entry_date": entry_date, "entry_price": entry_price,
                    "entry_code": "BUY", "exit_date": dates[i],
                    "exit_price": c[i], "exit_code": exit_code,
                    "bars_held": i - entry_idx,
                    "gross_return_pct": (c[i] / entry_price - 1.0) * 100.0,
                    "resolved": True,
                })

        buy = (
            i > 0 and c[i] > o[i] and c[i] > c[i - 1]
            and mid[i] >= mid_prev[i]
            and c[i] > sma5[i] and c[i] < mid[i]
        )
        # A buy can follow an exit on the same bar, as in the Pine.
        if not in_trade and buy:
            in_trade, entry_price, entry_date, entry_idx = True, c[i], dates[i], i
            cl_price = max(low7[i], entry_price * (1.0 - MAX_CL_PCT / 100.0))
            codes[i] = "BUY" if not codes[i] else f"{codes[i]} + BUY"

        position_arr[i] = int(in_trade)

    if in_trade:
        last = n - 1
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": "BUY", "exit_date": dates[last],
            "exit_price": c[last], "exit_code": "OPEN",
            "bars_held": last - entry_idx,
            "gross_return_pct": (c[last] / entry_price - 1.0) * 100.0,
            "resolved": False,
        })

    # No armed-but-not-bought state in this Pine, so never a watchlist row.
    return codes, trades, {"position": position_arr, "setup": [False] * n}


if __name__ == "__main__":
    # Self-check, no store needed: a steady uptrend (rising mid), a pullback
    # under the mid, a green reclaim of SMA5 (BUY), then a rally that tags the
    # upper band on a red bar (TP). A second dip then breaks the stop (CL).
    close = pd.Series(
        list(np.linspace(100, 130, 40))                  # rising mid
        + [126, 122, 119, 117, 116, 115]                 # pull back under the mid
        + [118]                                          # green reclaim of SMA5 -> BUY
        + [122, 128, 136, 144]                           # rally to the upper band
        + [141]                                          # red bar tagging the band -> TP
        + [138, 134, 130, 126, 124, 123]                 # back under the mid
        + [126.2]                                        # BUY again
        + [100],                                         # crash through the stop -> CL
        dtype=float,
    )
    op = close.shift(1).fillna(close.iloc[0])
    op.iloc[46], op.iloc[58] = 114, 124                  # the two buy bars are green
    op.iloc[51] = 146                                    # the TP bar is red
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(close), freq="D"),
        "open": op, "close": close,
        "high": np.maximum(op, close) + 0.5, "low": np.minimum(op, close) - 0.5,
    })
    mid = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std(ddof=0)
    frame["ma5"] = close.rolling(5, min_periods=5).mean()
    frame["ma20"] = mid
    frame["bb_upper"] = mid + 2 * sd
    frame["high"] = np.where(frame.index == 51, frame["bb_upper"] + 1, frame["high"])
    codes, trades, _ = run_state_machine(prepare(frame))
    fired = [x for x in codes if x]
    assert fired[:2] == ["BUY", "TP"], fired
    assert "CL" in fired, fired
    tps = [t for t in trades if t["exit_code"] == "TP"]
    assert tps and all(t["gross_return_pct"] > 0 for t in tps), trades
    cl = next(t for t in trades if t["exit_code"] == "CL")
    assert cl["gross_return_pct"] < 0, cl
    print("pullback_sniper self-check ok:", [(i, x) for i, x in enumerate(codes) if x])
