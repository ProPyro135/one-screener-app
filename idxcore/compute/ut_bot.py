"""UT Bot strategy (ATR trailing stop 1 x ATR100, doji entry, fixed -5% CL).

A port of the Pine Script "UT Bot Fixed 1:100 (Doji Entry & No Buy Above)", in
the same shape as ``market_structure`` so ``trade_log`` serves it unchanged.

    BUY      a bar that opens at or under the trailing stop and closes green or
             flat, either through the stop or, still under it, as a symmetric
             doji (body <= 25% of the range, each wick >= 25%)
    TP UT    close crosses under the trailing stop, above the entry price
    TP BB    the high tags the upper Bollinger band on a red bar, above entry
    CL -5%   close at or below 95% of the entry price

The trailing stop is the Pine's own recursion over ATR(100) (Wilder RMA, as
Pine's ``atr``). The upper band is read from the stored indicators (population
stdev, as Pine), not recomputed. Fills are at the signal bar's close, like the
other strategies; the Pine's TPs both require the close to be above the entry,
so every TP is a profit.

A *monitoring* screener with no validated edge; bar colours, labels and alerts
are visual only and are not ported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idxcore.compute.bottom_fishing import _all_bars, _ticker_meta  # noqa: F401
from idxcore.compute.reversal_sniper import _rma

KEY_VALUE = 1.0
ATR_PERIOD = 100
CL_PCT = 5.0
DOJI_BODY = 0.25
DOJI_WICK = 0.25

MIN_BARS = ATR_PERIOD + 1  # the stop is undefined until ATR(100) is

CATEGORY = {"BUY": "BUY", "TP": "TAKE PROFIT", "CL": "CUT LOSS"}


def category_of(code: str) -> str:
    return CATEGORY.get(code.split(" ", 1)[0], "OTHER")


def _trailing_stop(c: np.ndarray, n_loss: np.ndarray) -> np.ndarray:
    """The Pine ``xATRTrailingStop`` recursion; NaN until ATR exists."""
    stop = np.full(len(c), np.nan)
    for i in range(len(c)):
        if np.isnan(n_loss[i]):
            continue
        prev = 0.0 if i == 0 or np.isnan(stop[i - 1]) else stop[i - 1]  # nz(stop[1], 0)
        prev_c = c[i - 1] if i > 0 else np.nan
        if c[i] > prev and prev_c > prev:
            stop[i] = max(prev, c[i] - n_loss[i])
        elif c[i] < prev and prev_c < prev:
            stop[i] = min(prev, c[i] + n_loss[i])
        elif c[i] > prev:
            stop[i] = c[i] - n_loss[i]
        else:
            stop[i] = c[i] + n_loss[i]
    return stop


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    """Add ATR(100) and the trailing stop."""
    df = history.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    df["atr"] = _rma(tr, ATR_PERIOD)
    df["ut_stop"] = _trailing_stop(close.to_numpy(float), (KEY_VALUE * df["atr"]).to_numpy(float))
    return df


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Entry first, then the exit check, in the Pine's own order."""
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    upper = pd.to_numeric(df["bb_upper"], errors="coerce").to_numpy(float)
    stop = df["ut_stop"].to_numpy(float)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n

    in_trade = False
    entry_price = np.nan
    entry_date = None
    entry_idx = -1

    for i in range(n):
        rng = h[i] - lo[i]
        body = abs(c[i] - o[i])
        doji = (
            rng > 0 and body <= rng * DOJI_BODY
            and h[i] - max(c[i], o[i]) >= rng * DOJI_WICK
            and min(c[i], o[i]) - lo[i] >= rng * DOJI_WICK
        )
        raw_buy = (
            o[i] <= stop[i] and c[i] >= o[i]
            and (c[i] > stop[i] or (c[i] <= stop[i] and doji))
        )
        if raw_buy and not in_trade:
            in_trade, entry_price, entry_date, entry_idx = True, c[i], dates[i], i
            codes[i] = "BUY"

        # Checked on the entry bar too, as in the Pine; nothing can fire there,
        # since every exit compares the close with the entry price.
        if in_trade:
            crossunder = i > 0 and c[i] < stop[i] and c[i - 1] >= stop[i - 1]
            exit_code = ""
            if crossunder and c[i] > entry_price:
                exit_code = "TP UT"
            elif h[i] >= upper[i] and c[i] < o[i] and c[i] > entry_price:
                exit_code = "TP BB"
            elif c[i] <= entry_price * (1.0 - CL_PCT / 100.0):
                exit_code = "CL -5%"
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
    # Self-check, no store needed: a slide keeps the stop above price; a green
    # bar opening under it and closing through it buys; a rally then a close
    # back under the stop takes profit; a second buy then drops 5% and cuts.
    close = list(np.linspace(200, 100, 120))          # stop rides above
    close += [104]                                     # 120: BUY through the stop
    close += list(np.linspace(106, 130, 12))           # 121-132: rally, stop trails under
    close += [118]                                     # 133: close back under -> TP UT
    close += list(np.linspace(116, 100, 10))           # 134-143: slide, stop above again
    close += [104]                                     # 144: BUY again
    close += [98]                                      # 145: -5.8% -> CL
    close = pd.Series(close, dtype=float)
    op = close.shift(1).fillna(close.iloc[0])
    op.iloc[120], op.iloc[144] = 99, 99                # buy bars open under the stop
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(close), freq="D"),
        "open": op, "close": close,
        "high": np.maximum(op, close) + 1, "low": np.minimum(op, close) - 1,
        "bb_upper": np.inf,                            # keep the BB exit out of this path
    })
    codes, trades, lines = run_state_machine(prepare(frame))
    fired = [(i, x) for i, x in enumerate(codes) if x]
    assert [x for _, x in fired] == ["BUY", "TP UT", "BUY", "CL -5%"], fired
    tp, cl = trades[0], trades[1]
    assert tp["gross_return_pct"] > 0 and cl["gross_return_pct"] <= -CL_PCT, trades
    assert all(p in (0, 1) for p in lines["position"])
    print("ut_bot self-check ok:", fired)
