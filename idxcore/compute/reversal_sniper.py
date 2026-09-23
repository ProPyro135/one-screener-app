"""Reversal Sniper strategy (HH-HL breakout + MA20 re-entry).

A port of the Pine Script "Reversal Sniper: HH-HL + Re-Entry MA20", in the
same shape as ``market_structure`` — a per-ticker state machine returning
``(codes, trades, lines)`` — so ``trade_log`` serves it unchanged.

    BUY HH-HL       a 50-bar low (L1), a rally high (H1), a higher low (L2),
                    then a close above H1 and MA5
    BUY Re-Entry    after a TP: close dips under MA20, then crosses back above
    TP Smart MA5    once price has flown >= 1 ATR above MA5 (with close above
                    MA20), the first close back under MA5 that is still above
                    the entry price
    CL Struktur     the bar's low touches the stop: L2 - 1% (main entry) or
                    the 7-bar low - 1% (re-entry)

Fills are at the signal bar's close, like ``market_structure`` and the Pine's
own ``entryPrice := close``. The stop triggers on the bar's low but the exit is
recorded at that bar's close, so a CL can close above or below the stop.

A *monitoring* screener with no validated edge; the Pine's lines, labels and
plots are visual only and are not ported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idxcore.compute.market_structure import _all_bars, _pivot_series, _ticker_meta  # noqa: F401

# Pine inputs and constants (defaults)
BOTTOM_LOOKBACK = 50
PIVOT_LEN = 4
TP_ATR_MULT = 1.0
ATR_LEN = 14
REENTRY_LOW_BARS = 7
CL_BUFFER = 0.99

MIN_BARS = BOTTOM_LOOKBACK  # is_bottom needs a full 50-bar window

CATEGORY = {"BUY": "BUY", "TP": "TAKE PROFIT", "CL": "CUT LOSS"}


def category_of(code: str) -> str:
    return CATEGORY.get(code.split(" ", 1)[0], "OTHER")


def _rma(x: pd.Series, n: int) -> pd.Series:
    """Pine ``ta.rma``: seeded with the SMA of the first ``n`` values."""
    v = x.to_numpy(float)
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        out[n - 1] = v[:n].mean()
        for i in range(n, len(v)):
            out[i] = (out[i - 1] * (n - 1) + v[i]) / n
    return pd.Series(out, index=x.index)


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    """Add MA5/MA20, ATR(14), the 50-bar bottom flag, 7-bar low and pivot lows."""
    df = history.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")

    df["ma5"] = close.rolling(5, min_periods=5).mean()
    df["ma20"] = close.rolling(20, min_periods=20).mean()
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    df["atr"] = _rma(tr, ATR_LEN)
    df["is_bottom"] = low == low.rolling(BOTTOM_LOOKBACK, min_periods=BOTTOM_LOOKBACK).min()
    df["low7"] = low.rolling(REENTRY_LOW_BARS, min_periods=REENTRY_LOW_BARS).min()
    df["pivot_low"] = _pivot_series(low.to_numpy(float), PIVOT_LEN, PIVOT_LEN, False)
    return df


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Walk the bars top to bottom in the Pine's own order."""
    dates = df["date"].to_numpy()
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    ma5 = df["ma5"].to_numpy(float)
    ma20 = df["ma20"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    low7 = df["low7"].to_numpy(float)
    is_bottom = df["is_bottom"].to_numpy(bool)
    pl = df["pivot_low"].to_numpy(float)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n
    setup_arr = [False] * n

    state = 0
    L1 = H1 = L2 = np.nan
    H1_bar = -1
    in_trade = False
    entry_price = np.nan
    entry_date = None
    entry_code = ""
    entry_idx = -1
    cl_price = np.nan
    siap_tp = False
    can_reentry = False
    dropped = False

    for i in range(n):
        # 2. structure tracker
        if is_bottom[i]:
            L1, H1, H1_bar = lo[i], h[i], i
            state = 1
        buy_signal = False
        if state == 1:
            if h[i] > H1:
                H1, H1_bar = h[i], i
            if not np.isnan(pl[i]):
                if i - PIVOT_LEN > H1_bar and pl[i] > L1:
                    L2 = pl[i]
                    state = 2
        elif state == 2:
            if not np.isnan(pl[i]) and i - PIVOT_LEN > H1_bar and pl[i] > L1:
                L2 = pl[i]
            if lo[i] < L1:
                state = 0
            # not an elif in the Pine: a breakout on the same bar still fires
            if c[i] > H1 and c[i] > ma5[i]:
                buy_signal = True
                state = 3

        # 4. re-entry watch
        reentry = False
        if can_reentry and not in_trade:
            if c[i] < ma20[i]:
                dropped = True
            crossover = (i > 0 and c[i] > ma20[i] and c[i - 1] <= ma20[i - 1])
            if dropped and crossover:
                reentry = True

        # 5. entry
        if (buy_signal or reentry) and not in_trade:
            in_trade, entry_price, siap_tp = True, c[i], False
            entry_date, entry_idx = dates[i], i
            if buy_signal:
                entry_code = "BUY HH-HL"
                cl_price = L2 * CL_BUFFER
            else:
                entry_code = "BUY Re-Entry"
                cl_price = low7[i] * CL_BUFFER
                can_reentry = dropped = False
            codes[i] = entry_code

        # exit, evaluated on the entry bar too, as in the Pine
        if in_trade:
            if c[i] > ma20[i] and (h[i] - ma5[i]) >= atr[i] * TP_ATR_MULT:
                siap_tp = True
            exit_code = ""
            if lo[i] <= cl_price:
                exit_code = "CL Struktur"
                can_reentry = dropped = False
            # Owner's rule: a take-profit must be a profit. Not in the original
            # Pine; otherwise the position stays open until a real TP or the CL.
            elif siap_tp and c[i] < ma5[i] and c[i] > entry_price:
                exit_code = "TP Smart MA5"
                can_reentry, dropped = True, False
            if exit_code:
                in_trade = False
                codes[i] = exit_code
                trades.append({
                    "entry_date": entry_date, "entry_price": entry_price,
                    "entry_code": entry_code, "exit_date": dates[i],
                    "exit_price": c[i], "exit_code": exit_code,
                    "bars_held": i - entry_idx,
                    "gross_return_pct": (c[i] / entry_price - 1.0) * 100.0,
                    "resolved": True,
                })

        position_arr[i] = int(in_trade)
        # Armed with nothing bought: a HH-HL structure waiting for its
        # breakout. The post-TP re-entry watch is left out, or every TP would
        # show as WATCHLIST instead of CLOSED.
        setup_arr[i] = (not in_trade) and state == 2

    if in_trade:
        last = n - 1
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": entry_code, "exit_date": dates[last],
            "exit_price": c[last], "exit_code": "OPEN",
            "bars_held": last - entry_idx,
            "gross_return_pct": (c[last] / entry_price - 1.0) * 100.0,
            "resolved": False,
        })

    return codes, trades, {"position": position_arr, "setup": setup_arr}


if __name__ == "__main__":
    # Self-check, no store needed: a slide to a 50-bar low, a rally, a higher
    # low, a breakout (BUY HH-HL), a spike far above MA5 and a close back
    # under it (TP), then a dip under MA20 and a reclaim (BUY Re-Entry).
    close = np.array(
        list(np.linspace(200, 100, 60))                 # slide to the bottom
        + [104, 108, 112, 116, 120]                     # rally -> H1 ~ 120.5
        + [117, 114, 111, 108, 110, 113, 115, 116]      # higher low at 108
        + [118, 122, 126]                               # breakout over H1
        + [140, 150, 160]                               # fly far above MA5
        + [150, 138, 128, 118, 112, 108, 106]           # back under MA5, then MA20
        + [115, 124, 130],                              # reclaim MA20
        dtype=float,
    )
    n = len(close)
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
    })
    codes, trades, lines = run_state_machine(prepare(frame))
    fired = [x for x in codes if x]
    assert "BUY HH-HL" in fired, fired
    assert "TP Smart MA5" in fired, fired
    assert fired.index("BUY HH-HL") < fired.index("TP Smart MA5"), fired
    assert "BUY Re-Entry" in fired, fired
    assert all(t["gross_return_pct"] > 0 for t in trades if t["exit_code"] == "TP Smart MA5"), trades
    assert all(p in (0, 1) for p in lines["position"])
    print("reversal_sniper self-check ok:", [(i, x) for i, x in enumerate(codes) if x])
