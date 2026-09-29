"""Sniper VCP strategy (strategy I, EOD).

A port of the Pine Script "Market Structure: Sniper VCP".

    Base           the prior 40 bars span <= 12% of their low, in a Minervini
                   stage 2 (close > EMA50 > EMA150 > EMA200)
    BUY Sniper     a close crossing over the prior 20-bar high, the bar after
                   a base bar, on volume >= 2x its 50-bar average, closing in
                   the top 30% of its bar
    First stop     the higher of the base low and the signal close - 5.5%
    Trailing       once the highest high since the signal reaches the buy
                   price + 5%, the stop is at least that high - 3%, only rising
    TP             the whole position at the buy price + 20%
    Time stop      10 bars after the signal, if the trailing never switched
                   on, sold at the next open

The stop and the TP are resting orders from the bar after the fill, filled at
their level or at the open on a gap through; a bar that reaches both counts
as the stop.

As written the Pine never exits: its ``else`` branch (no position yet) wipes
the entry price, the stop, the running high and the entry bar on the signal
bar, and all of them are only set on a signal bar, so every exit stays
``na``. Ported as its comments intend instead, with the buy price taken as the
actual fill (the next open) so a TP is always over the price paid. The Pine's
20% sizing and 0.2% commission are not ported; the page's fee model applies.

A sale over the buy price counts as TP, otherwise as CL.

A *monitoring* screener with no validated edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idxcore.compute.early_entry import _all_bars, _ticker_meta, category_of  # noqa: F401

ACCUM_LEN, MICRO_LEN, MAX_BOX_PCT, VOL_MULT = 40, 20, 12.0, 2.0
TP_PCT = 20.0
MAX_SL_PCT = 5.5
TRAIL_ON_PCT, TRAIL_OFFSET_PCT = 5.0, 3.0
MAX_HOLD = 10

MIN_BARS = 201  # EMA200


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    """Stage 2 EMAs, the 40-bar base, the 20-bar breakout and volume."""
    df = history.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")

    ema = {n: close.ewm(span=n, adjust=False, min_periods=n).mean() for n in (50, 150, 200)}
    stage2 = (close > ema[50]) & (ema[50] > ema[150]) & (ema[150] > ema[200])
    range_high = high.shift(1).rolling(ACCUM_LEN, min_periods=ACCUM_LEN).max()
    range_low = low.shift(1).rolling(ACCUM_LEN, min_periods=ACCUM_LEN).min()
    accum = ((range_high - range_low) / range_low * 100 <= MAX_BOX_PCT) & stage2
    micro = high.shift(1).rolling(MICRO_LEN, min_periods=MICRO_LEN).max()
    cross = (close > micro) & (close.shift(1) <= micro.shift(1))
    vol_ok = vol >= vol.rolling(50, min_periods=50).mean() * VOL_MULT
    strong = close >= low + (high - low) * 0.7
    df["signal"] = cross & accum.shift(1, fill_value=False) & vol_ok & strong
    df["range_low"] = range_low
    return df


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Signals at the close, fills at the next open, TP and stop resting."""
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    signal = df["signal"].to_numpy(bool)
    range_low = df["range_low"].to_numpy(float)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n

    in_pos = pending_buy = orders = trailing = False
    pending_sell = ""
    entry_price = stop = highest = np.nan
    entry_date = None
    entry_idx = signal_idx = -1

    def close_trade(i: int, why: str, price: float) -> None:
        code = ("TP " if price > entry_price else "CL ") + why
        codes[i] = code
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": "BUY Sniper", "exit_date": dates[i],
            "exit_price": price, "exit_code": code, "bars_held": i - entry_idx,
            "gross_return_pct": (price / entry_price - 1.0) * 100.0, "resolved": True,
        })

    for i in range(n):
        # 1. Orders from the previous close fill at this open.
        if pending_sell:
            close_trade(i, pending_sell, o[i])
            in_pos, pending_sell = False, ""
        if pending_buy:
            pending_buy, in_pos, orders, trailing = False, True, False, False
            entry_price, entry_date, entry_idx = o[i], dates[i], i
            codes[i] = "BUY Sniper"
        # 2. Resting exits, placed at an earlier close. The stop first.
        if in_pos and orders:
            tp = entry_price * (1 + TP_PCT / 100)
            if lo[i] <= stop:
                in_pos = False
                close_trade(i, "Trailing stop" if trailing else "Stop loss", min(o[i], stop))
            elif h[i] >= tp:
                in_pos = False
                close_trade(i, f"Target {TP_PCT:g}%", max(o[i], tp))
        # 3. At the close.
        if in_pos:
            highest = max(highest, h[i])
            if highest >= entry_price * (1 + TRAIL_ON_PCT / 100):
                trailing = True
                stop = max(stop, highest * (1 - TRAIL_OFFSET_PCT / 100))
            orders = True
            if not trailing and i - signal_idx >= MAX_HOLD:
                pending_sell = "Time stop"
        elif signal[i]:
            pending_buy, signal_idx, highest = True, i, c[i]
            stop = max(range_low[i], c[i] * (1 - MAX_SL_PCT / 100))
        position_arr[i] = int(in_pos)

    if in_pos:
        last = n - 1
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": "BUY Sniper", "exit_date": dates[last],
            "exit_price": c[last], "exit_code": "OPEN", "bars_held": last - entry_idx,
            "gross_return_pct": (c[last] / entry_price - 1.0) * 100.0, "resolved": False,
        })

    # A buy signalled at the last close, to be filled at tomorrow's open.
    setup_arr = [False] * n
    if n and pending_buy:
        setup_arr[-1] = True
    return codes, trades, {"position": position_arr, "setup": setup_arr}


if __name__ == "__main__":
    n = 20

    def frame():
        f = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n, "close": [100.0] * n,
            "signal": [False] * n, "range_low": [90.0] * n,
        })
        f.loc[1, "signal"] = True
        return f

    # First stop = max(90, 94.5) = 94.5; bar 3 dips to 93 -> CL at 94.5.
    f = frame()
    f.loc[3, "low"] = 93.0
    _, t, _ = run_state_machine(f)
    assert (t[0]["exit_code"], t[0]["exit_price"]) == ("CL Stop loss", 94.5), t[0]
    # Trailing: bar 3 reaches 108 -> stop 104.76; bar 4 dips to 104 -> TP at 104.76.
    f = frame()
    f.loc[3, "high"] = 108.0
    f.loc[4, "low"] = 104.0
    f.loc[4, "open"] = 106.0
    _, t, _ = run_state_machine(f)
    assert t[0]["exit_code"] == "TP Trailing stop" and round(t[0]["exit_price"], 2) == 104.76, t[0]
    # TP: bar 3 reaches 121 -> sold at 120.
    f = frame()
    f.loc[3, "high"] = 121.0
    _, t, _ = run_state_machine(f)
    assert t[0]["exit_code"] == "TP Target 20%" and round(t[0]["exit_price"], 6) == 120.0, t[0]
    # Time stop: flat -> decided at bar 11 (10 after the signal), sold at bar 12's open.
    _, t, _ = run_state_machine(frame())
    assert t[0]["exit_code"] == "CL Time stop" and pd.Timestamp(t[0]["exit_date"]) == frame().loc[12, "date"], t[0]
    print("sniper_vcp self-check ok")
