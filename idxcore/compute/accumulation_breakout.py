"""Accumulation Breakout & Trend Run strategy (EOD).

A port of the Pine Script "Market Structure: Accumulation Breakout & Trend
Run" at its defaults, in the same shape as ``market_structure`` so
``trade_log`` serves it unchanged.

    Accumulation   the prior 25 bars' high-low range spans <= 15% of its low,
                   with the close >= 0.95x EMA50
    BUY Breakout   a close crossing over the prior 25-bar high, the bar after
                   an accumulation bar, on volume >= 1.3x its 25-bar average
    Trailing stop  a Chandelier exit: the highest high since the BUY minus
                   3 ATR(14), only ever rising; a close under it sells at the
                   next open

The Pine has no take-profit, only the trailing stop. The page's statuses need
one, so the exit is split by where it fills: over the buy price it is a TP
("TP Trailing stop"), otherwise a CL ("CL Trailing stop").

Execution follows the Pine with ``process_orders_on_close = false``: the BUY
fills at the next open, and so does the sell after a close under the stop. A
signal on the last bar has not been filled yet: it shows as a WATCHLIST row.
The Pine's all-in sizing and 0.15% commission are not ported; the page's fee
model (0.15% buy, 0.25% sell) applies instead.

A *monitoring* screener with no validated edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from idxcore.compute.market_structure import _all_bars, _ticker_meta  # noqa: F401
from idxcore.compute.reversal_sniper import _rma

ATR_LEN = 14


@dataclass(frozen=True)
class Params:
    """One Pine variant of the accumulation breakout."""
    accum_len: int           # bars in the base (the prior N bars' range)
    max_box_pct: float       # widest base, % of its low
    vol_len: int             # volume average length
    vol_mult: float          # breakout volume, x that average
    ema_len: int             # trend EMA
    ema_floor: float         # base needs close >= floor x EMA (> when strict)
    ema_strict: bool
    atr_mult: float          # Chandelier: peak minus N ATR
    strong_candle: bool      # breakout must close in the top 30% of its bar
    stop_at_base_low: bool   # first stop: max(close - N ATR, base low)


#: PINESCRIPT D, "Market Structure: Accumulation Breakout & Trend Run".
D = Params(accum_len=25, max_box_pct=15.0, vol_len=25, vol_mult=1.3, ema_len=50,
           ema_floor=0.95, ema_strict=False, atr_mult=3.0, strong_candle=False,
           stop_at_base_low=False)

MIN_BARS = D.ema_len + 1

CATEGORY = {"BUY": "BUY", "TP": "TAKE PROFIT", "CL": "CUT LOSS"}


def category_of(code: str) -> str:
    return CATEGORY.get(code.split(" ", 1)[0], "OTHER")


def prepare(history: pd.DataFrame, p: Params = D) -> pd.DataFrame:
    """The range, ATR, volume average and EMA the Pine reads."""
    df = history.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")

    range_high = high.shift(1).rolling(p.accum_len, min_periods=p.accum_len).max()
    range_low = low.shift(1).rolling(p.accum_len, min_periods=p.accum_len).min()
    span = (range_high - range_low) / range_low * 100
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    df["atr"] = _rma(tr, ATR_LEN)
    ema = close.ewm(span=p.ema_len, adjust=False, min_periods=p.ema_len).mean()
    above = close > ema * p.ema_floor if p.ema_strict else close >= ema * p.ema_floor
    accum = (span <= p.max_box_pct) & above
    cross = (close > range_high) & (prev <= range_high.shift(1))
    vol_ok = vol >= vol.rolling(p.vol_len, min_periods=p.vol_len).mean() * p.vol_mult
    signal = cross & accum.shift(1, fill_value=False) & vol_ok
    if p.strong_candle:
        signal &= close >= low + (high - low) * 0.7
    df["signal"] = signal
    df["range_low"] = range_low
    return df


def run_state_machine(df: pd.DataFrame, p: Params = D) -> tuple[list[str], list[dict], dict]:
    """Signals at the close, fills at the next open."""
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    atr = df["atr"].to_numpy(float)
    signal = df["signal"].to_numpy(bool)
    range_low = df["range_low"].to_numpy(float)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n

    in_pos = pending_buy = pending_sell = False
    entry_price = trail = highest = np.nan
    entry_date = None
    entry_idx = -1

    for i in range(n):
        # 1. Orders from the previous close fill at this open.
        if pending_sell:
            pending_sell, in_pos = False, False
            code = "TP Trailing stop" if o[i] > entry_price else "CL Trailing stop"
            codes[i] = code
            trades.append({
                "entry_date": entry_date, "entry_price": entry_price,
                "entry_code": "BUY Breakout", "exit_date": dates[i],
                "exit_price": o[i], "exit_code": code,
                "bars_held": i - entry_idx,
                "gross_return_pct": (o[i] / entry_price - 1.0) * 100.0,
                "resolved": True,
            })
        if pending_buy:
            pending_buy, in_pos = False, True
            entry_price, entry_date, entry_idx = o[i], dates[i], i
            codes[i] = "BUY Breakout" if not codes[i] else f"{codes[i]} + BUY Breakout"
        # 2. At the close. The stop starts at the signal (its close minus
        #    N ATR, the peak at its high) and only rises from there.
        if in_pos and not pending_sell:
            highest = max(highest, h[i])
            trail = max(trail, highest - atr[i] * p.atr_mult)
            if c[i] < trail:
                pending_sell = True
        elif not in_pos and signal[i]:
            pending_buy = True
            highest, trail = h[i], c[i] - atr[i] * p.atr_mult
            if p.stop_at_base_low:
                trail = max(trail, range_low[i])
        position_arr[i] = int(in_pos)

    if in_pos:
        last = n - 1
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": "BUY Breakout", "exit_date": dates[last],
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
    # A flat 30-bar base (range ~2%), a volume breakout, a run up, then a
    # slide: the Chandelier stop follows the run and sells on the slide.
    base = [100.0 + (i % 2) for i in range(30)]
    close = np.array(base + [110, 115, 120, 126, 132, 138, 140, 139, 130, 118, 110], dtype=float)
    n = len(close)
    vol = np.full(n, 1000.0)
    vol[30] = 5000.0
    f = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close - 0.5, "high": close + 1, "low": close - 1, "close": close, "volume": vol,
    })
    # prepare's EMA50 needs 50 bars; this frame is shorter, so feed it directly.
    df = prepare(f)
    df["signal"] = False
    df.loc[30, "signal"] = True
    codes, trades, lines = run_state_machine(df)
    fired = [(i, x) for i, x in enumerate(codes) if x]
    assert fired[0] == (31, "BUY Breakout"), fired
    assert len(trades) == 1 and trades[0]["resolved"], trades
    assert trades[0]["entry_price"] == 114.5, trades[0]
    assert trades[0]["exit_code"] == "TP Trailing stop" and trades[0]["exit_price"] > 114.5, trades[0]
    assert not any(lines["setup"])
    # the signal logic itself: the base and the volume breakout are seen
    long = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=80, freq="D"),
        "close": [100.0 + (i % 2) for i in range(79)] + [106.0],
        "volume": [1000.0] * 79 + [5000.0],
    })
    long["open"], long["high"], long["low"] = long["close"] - 0.5, long["close"] + 1, long["close"] - 1
    assert prepare(long)["signal"].tolist() == [False] * 79 + [True]
    print("accumulation_breakout self-check ok:", fired)
