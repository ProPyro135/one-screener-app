"""UT Bot strategy (1 x ATR100 trailing stop, strict MA20 filter, run profit).

A port of the Pine Script "UT Bot Fixed 1:100 (Strict MA20 Filter & Run
Profit)", in the same shape as ``market_structure`` so ``trade_log`` serves it
unchanged.

    BUY      a green or flat bar that opens at or under yesterday's trailing
             stop and closes through today's, or stays under it as a symmetric
             doji (body <= 25% of the range, each wick >= 25%) within 3% of it.
             Blocked while MA20 sits below its value 10 bars ago, and after a
             TP while the close is under MA20 with more red than green bars in
             the last five.
    TP MA20  once the close has been above MA20 (at the buy or since), the
             first close back under MA20 — above the buy price
    CL -5%   a close at or below 95% of the buy price

Owner's rule: a take-profit must be a profit. The Pine takes TP on any close
under MA20 after touching it, so it can "take profit" below the buy price;
here that close does not exit, and the trade runs on to a TP above the entry
or the CL.

The trailing stop is the Pine's own recursion over ATR(100) (Wilder RMA, as
Pine's ``atr``); MA20 is read from the stored indicators. Fills are at the
signal bar's close, like the other strategies.

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
NEAR_LINE = 0.97        # a doji under the stop must close within 3% of it
MA20_SLOPE_BARS = 10
COLOUR_BARS = 5

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
    """Add ATR(100), the trailing stop, MA20's value 10 bars back and the
    red/green bar counts over the last five bars."""
    df = history.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    df["atr"] = _rma(tr, ATR_PERIOD)
    df["ut_stop"] = _trailing_stop(close.to_numpy(float), (KEY_VALUE * df["atr"]).to_numpy(float))
    df["ma20_prev"] = pd.to_numeric(df["ma20"], errors="coerce").shift(MA20_SLOPE_BARS)
    red = (close < open_).astype(float)
    df["red5"] = red.rolling(COLOUR_BARS, min_periods=COLOUR_BARS).sum()
    df["green5"] = (1.0 - red).rolling(COLOUR_BARS, min_periods=COLOUR_BARS).sum()
    return df


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Entry, then the exit check on later bars, in the Pine's own order."""
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    ma20 = pd.to_numeric(df["ma20"], errors="coerce").to_numpy(float)
    ma20_prev = df["ma20_prev"].to_numpy(float)
    red5 = df["red5"].to_numpy(float)
    green5 = df["green5"].to_numpy(float)
    stop = df["ut_stop"].to_numpy(float)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n

    in_trade = False
    entry_price = np.nan
    entry_date = None
    entry_idx = -1
    touched_ma20 = False
    last_exit = ""

    for i in range(n):
        rng = h[i] - lo[i]
        doji = (
            rng > 0 and abs(c[i] - o[i]) <= rng * DOJI_BODY
            and h[i] - max(c[i], o[i]) >= rng * DOJI_WICK
            and min(c[i], o[i]) - lo[i] >= rng * DOJI_WICK
        )
        prev_stop = stop[i - 1] if i > 0 else np.nan
        raw_buy = (
            o[i] <= prev_stop and c[i] >= o[i]
            and (c[i] > stop[i]
                 or (c[i] <= stop[i] and c[i] >= stop[i] * NEAR_LINE and doji))
        )
        ma20_falling = ma20[i] < ma20_prev[i]
        no_rebuy_after_tp = c[i] < ma20[i] and last_exit == "TP" and red5[i] > green5[i]
        buy = raw_buy and not in_trade and not ma20_falling and not no_rebuy_after_tp

        if buy:
            in_trade, entry_price, entry_date, entry_idx = True, c[i], dates[i], i
            last_exit = ""
            touched_ma20 = c[i] > ma20[i]
            codes[i] = "BUY"
        elif in_trade:
            if c[i] > ma20[i]:
                touched_ma20 = True
            exit_code = ""
            if touched_ma20 and c[i] < ma20[i] and c[i] > entry_price:
                exit_code, last_exit = "TP MA20", "TP"
            elif c[i] <= entry_price * (1.0 - CL_PCT / 100.0):
                exit_code, last_exit = "CL -5%", "CL"
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
    # Self-check, no store needed: a steady climb (MA20 rising), a short dip
    # that puts the stop above price, a green bar opening under yesterday's
    # stop and closing through it (BUY), a run above MA20 and a close back
    # under it still above the entry (TP). A second dip-and-buy then falls 5%.
    close = list(np.linspace(100, 160, 130))           # 0-129: climb
    close += [152, 146, 142]                           # 130-132: dip, stop flips above
    close += [150]                                     # 133: BUY through the stop
    close += [156, 162, 166, 168, 158]                 # 134-138: run over MA20
    close += [166, 170, 174, 176, 178, 180]            # 139-144: still over it
    close += [170, 164, 160]                           # 145-147: 147 closes under MA20,
                                                       # above the entry -> TP
    close += [168]                                     # 148: BUY
    close += [158]                                     # 149: -6% -> CL
    close = pd.Series(close, dtype=float)
    op = close.shift(1).fillna(close.iloc[0])
    op.iloc[133], op.iloc[148] = 141, 159              # buy bars open under the stop
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(close), freq="D"),
        "open": op, "close": close,
        "high": np.maximum(op, close) + 1, "low": np.minimum(op, close) - 1,
        "ma20": close.rolling(20, min_periods=20).mean(),
    })
    codes, trades, lines = run_state_machine(prepare(frame))
    fired = [(i, x) for i, x in enumerate(codes) if x]
    assert [x for _, x in fired] == ["BUY", "TP MA20", "BUY", "CL -5%"], fired
    tp, cl = trades[0], trades[1]
    assert tp["gross_return_pct"] > 0 and cl["gross_return_pct"] <= -CL_PCT, trades
    assert all(p in (0, 1) for p in lines["position"])
    print("ut_bot self-check ok:", fired)
