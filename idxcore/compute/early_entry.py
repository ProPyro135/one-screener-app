"""Early Entry & Hard TP strategy (EOD).

A port of the Pine Script "Market Structure: Early Entry & Hard TP" at its
defaults, in the same shape as ``market_structure`` so ``trade_log`` serves it
unchanged.

    Base           the prior 20 bars span <= 15% of their low, close > EMA100
    BUY Early      a close crossing over the prior 10-bar high, the bar after a
                   base bar, on volume >= 1.5x its 50-bar average, closing in
                   the top 40% of its bar
    TP Target 10%  a resting limit at the buy price + 10%
    Stop           the highest high since the fill minus 3 ATR(14), only ever
                   rising; a resting stop

Execution follows the Pine with ``process_orders_on_close = false``: the BUY
fills at the next open, and the exit orders are placed at that bar's close, so
they work from the bar after the fill, filled at their level or at the open on
a gap through. When one bar reaches both, the stop is taken (never resolve a
same-bar tie in the trade's favour).

The Pine sets the first stop at the base low on the signal bar, but its
``else`` branch (no position yet) clears it on the same bar, so in the Pine the
base-low stop never takes effect: the stop starts at the fill bar's high minus
3 ATR. Ported as the Pine runs, not as its comment reads.

A stop can trail above the buy price, so a stop exit over the buy counts as
TP ("TP Trailing stop"), otherwise CL ("CL Stop loss"). A signal on the last
bar has not been filled yet: it shows as a WATCHLIST row. The Pine's all-in
sizing and 0.15% commission are not ported; the page's fee model applies.

A *monitoring* screener with no validated edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idxcore.compute.market_structure import _all_bars, _ticker_meta  # noqa: F401
from idxcore.compute.reversal_sniper import _rma

ACCUM_LEN = 20
MICRO_LEN = 10
MAX_BOX_PCT = 15.0
VOL_MULT = 1.5
EMA_LEN = 100
TP_PCT = 10.0
ATR_MULT = 3.0

MIN_BARS = EMA_LEN + 1

CATEGORY = {"BUY": "BUY", "TP": "TAKE PROFIT", "CL": "CUT LOSS"}


def category_of(code: str) -> str:
    return CATEGORY.get(code.split(" ", 1)[0], "OTHER")


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    """The base, micro resistance, ATR, volume average and EMA the Pine reads."""
    df = history.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")

    range_high = high.shift(1).rolling(ACCUM_LEN, min_periods=ACCUM_LEN).max()
    range_low = low.shift(1).rolling(ACCUM_LEN, min_periods=ACCUM_LEN).min()
    span = (range_high - range_low) / range_low * 100
    micro = high.shift(1).rolling(MICRO_LEN, min_periods=MICRO_LEN).max()
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    df["atr"] = _rma(tr, 14)
    ema = close.ewm(span=EMA_LEN, adjust=False, min_periods=EMA_LEN).mean()
    accum = (span <= MAX_BOX_PCT) & (close > ema)
    cross = (close > micro) & (prev <= micro.shift(1))
    vol_ok = vol >= vol.rolling(50, min_periods=50).mean() * VOL_MULT
    strong = close >= low + (high - low) * 0.6
    df["signal"] = cross & accum.shift(1, fill_value=False) & vol_ok & strong
    df["range_low"] = range_low      # strategy I's first stop
    return df


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Signals at the close, fills at the next open, TP and stop resting."""
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    atr = df["atr"].to_numpy(float)
    signal = df["signal"].to_numpy(bool)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n

    in_pos = pending_buy = orders = False
    entry_price = stop = limit = highest = np.nan
    entry_date = None
    entry_idx = -1

    def close_trade(i: int, code: str, price: float) -> None:
        codes[i] = code if not codes[i] else f"{codes[i]} + {code}"
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": "BUY Early", "exit_date": dates[i],
            "exit_price": price, "exit_code": code,
            "bars_held": i - entry_idx,
            "gross_return_pct": (price / entry_price - 1.0) * 100.0,
            "resolved": True,
        })

    for i in range(n):
        # 1. The BUY from the previous close fills at this open.
        if pending_buy:
            pending_buy, in_pos, orders = False, True, False
            entry_price, entry_date, entry_idx = o[i], dates[i], i
            codes[i] = "BUY Early"
        # 2. Resting exits, placed at an earlier close. The stop first.
        if in_pos and orders:
            if lo[i] <= stop:
                px = min(o[i], stop)
                in_pos = False
                close_trade(i, "TP Trailing stop" if px > entry_price else "CL Stop loss", px)
            elif h[i] >= limit:
                in_pos = False
                close_trade(i, f"TP Target {TP_PCT:g}%", max(o[i], limit))
        # 3. At the close: re-place the exits, or look for a BUY.
        if in_pos:
            highest = h[i] if np.isnan(highest) else max(highest, h[i])
            new = highest - atr[i] * ATR_MULT
            stop = new if np.isnan(stop) else max(stop, new)
            limit = entry_price * (1 + TP_PCT / 100)
            orders = True
        else:
            highest = stop = np.nan
            if signal[i]:
                pending_buy = True
        position_arr[i] = int(in_pos)

    if in_pos:
        last = n - 1
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": "BUY Early", "exit_date": dates[last],
            "exit_price": c[last], "exit_code": "OPEN",
            "bars_held": last - entry_idx,
            "gross_return_pct": (c[last] / entry_price - 1.0) * 100.0,
            "resolved": False,
        })

    # A buy signalled at the last close, to be filled at tomorrow's open.
    setup_arr = [False] * n
    if n and pending_buy:
        setup_arr[-1] = True
    return codes, trades, {"position": position_arr, "setup": setup_arr}


if __name__ == "__main__":
    # Signal at bar 1, fill at bar 2's open (100); no exit on the fill bar even
    # though it spikes to 115; bar 4 reaches 110 -> TP at the limit. Signal at
    # bar 6, fill at 7 (100); bar 9 gaps down under the stop -> CL at the open.
    n = 12
    f = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": [100.0] * n, "high": [102.0] * n, "low": [98.0] * n, "close": [100.0] * n,
        "atr": [2.0] * n, "signal": [False] * n,
    })
    f.loc[[1, 6], "signal"] = True
    f.loc[2, "high"] = 115.0                    # fill bar: exits not placed yet
    f.loc[4, "high"] = 111.0                    # reaches the 110 limit
    f.loc[9, ["open", "low"]] = [90.0, 89.0]    # gap under the stop
    codes, trades, lines = run_state_machine(f)
    # bar 2's high 115 sets the stop at 115 - 6 = 109, above bar 3's low 98:
    # bar 3 stops out at its open 100 (not above the 100 buy) -> CL.
    assert trades[0]["exit_code"] == "CL Stop loss" and trades[0]["exit_price"] == 100.0, trades[0]
    assert pd.Timestamp(trades[0]["exit_date"]) == pd.Timestamp(f.loc[3, "date"]), trades[0]
    f.loc[2, "high"] = 102.0                    # now stop 96, limit 110
    codes, trades, lines = run_state_machine(f)
    assert [(t["exit_code"], round(t["exit_price"], 6)) for t in trades] == [
        ("TP Target 10%", 110.0), ("CL Stop loss", 90.0)], trades
    assert category_of(trades[0]["exit_code"]) == "TAKE PROFIT"
    assert not any(lines["setup"])
    print("early_entry self-check ok:", [(i, x) for i, x in enumerate(codes) if x])
