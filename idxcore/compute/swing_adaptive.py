"""IDX Adaptive Swing v4 strategy (regime-adaptive swing low -> swing high, EOD).

A port of the Pine Script "IDX Adaptive Swing [EOD] v4" at its default preset
("Sangat selektif", exit "Seimbang"), in the same shape as
``market_structure`` so ``trade_log`` serves it unchanged.

Regime, re-read on every bar from the last two confirmed swing pivots (5/5):
UPTREND on a higher high and higher low or a close above the last swing high;
DOWNTREND on a lower high and lower low or a close under the last swing low;
SIDEWAYS otherwise (and until two of each pivot exist).

    BUY            a green bar closing above yesterday's high, within 2 ATR of
                   the 5-bar low, on a liquid stock (20-bar average turnover
                   >= Rp2bn), deep enough for its regime:
                     UPTREND   range position <= 25%, EMA50 gap >= -2 ATR,
                               5-bar RSI low <= 40, 5-bar Stoch low <= 25
                     SIDEWAYS  <= 10%, >= 0 ATR, RSI <= 35, Stoch <= 20
                     DOWNTREND <= 10%, >= 0 ATR, RSI <= 35, Stoch <= 20
                   Not within 2 bars of the last exit.
    TP Swing High  Stoch(14,3) >= 80 on a red bar closing 1% over the entry
    CL Stop loss   the 5-bar low at the signal minus 2.5 ATR, floored to the tick

Execution follows the Pine exactly, since it is written for end-of-day data:
a signal at a close is filled at the next bar's open (``process_orders_on_close
= false``), so the BUY row carries the fill day and the fill open. The stop is
a resting order from the fill bar on, filled at its level or at the open on a
gap through. The TP is a market close at the next open, so it is profitable at
the signal but can fill under the entry on a gap down. A signal on the last bar
has not been filled yet: it shows as a WATCHLIST row (buy at tomorrow's open).

RSI(14) is the stored Wilder RSI (invariant 1); the rest is computed here. The
Pine's backtest start date and display options are not ported; its fee table
(0.15% buy, 0.25% sell) is the one the page's net P&L uses.

A *monitoring* screener with no validated edge.
"""

from __future__ import annotations

from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from idxcore.compute.market_structure import _pivot_series, _ticker_meta  # noqa: F401
from idxcore.compute.reversal_sniper import _rma

PIVOT = 5
EMA_LEN = 50
RANGE_LEN = 60
MIN_TURNOVER = 2e9          # Rp2bn average daily value
COOL = 2                    # bars to wait after an exit
NEAR_LOW_ATR = 2.0
SL_ATR = 2.5              # preset "Sangat selektif"
X_STOCH = 80.0              # exit mode "Seimbang"
MIN_PROFIT = 1.0            # % over entry before a swing-high sell

UP, SIDE, DOWN = 1, 2, 3
REGIME_NAME = {UP: "UPTREND", SIDE: "SIDEWAYS", DOWN: "DOWNTREND"}
# regime -> (max range position, min EMA gap in ATR, max RSI low, max Stoch low)
# (preset "Sangat selektif", the v4 default)
DEPTH = {UP: (0.25, -2.0, 40.0, 25.0), SIDE: (0.10, 0.0, 35.0, 20.0),
         DOWN: (0.10, 0.0, 35.0, 20.0)}

#: The TP is a market order filled at the next open, so a gap down can fill it
#: under the entry; trade_log's "a TP is a profit" check skips this strategy.
TP_FILLS_NEXT_OPEN = True

MIN_BARS = RANGE_LEN + 1  # the Pine's warm-up: bar_index >= max(60, 50 + 10)

CATEGORY = {"BUY": "BUY", "TP": "TAKE PROFIT", "CL": "CUT LOSS"}

_BARS_SQL = """
    SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume, i.rsi14
      FROM prices p
      JOIN tickers t ON t.ticker = p.ticker
      LEFT JOIN indicators i ON i.ticker = p.ticker AND i.date = p.date
     WHERE {where}
     ORDER BY p.ticker, p.date
"""


def _all_bars(con: duckdb.DuckDBPyConnection, tickers: Optional[list[str]]) -> pd.DataFrame:
    if tickers is None:
        return con.execute(_BARS_SQL.format(where="t.is_active")).df()
    ph = ", ".join("?" for _ in tickers)
    return con.execute(
        _BARS_SQL.format(where=f"p.ticker IN ({ph})"), [t.upper() for t in tickers]
    ).df()


def category_of(code: str) -> str:
    return CATEGORY.get(code.split(" ", 1)[0], "OTHER")


def floor_tick(p: float) -> float:
    """The Pine's ``f_floorTick``: round down to the IDX price fraction."""
    t = 1.0 if p < 200 else 2.0 if p < 500 else 5.0 if p < 2000 else 10.0 if p < 5000 else 25.0
    return float(np.floor(p / t) * t)


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    """Every indicator the Pine reads, plus the per-bar regime."""
    df = history.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")

    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    df["atr"] = _rma(tr, 14)
    df["ema"] = close.ewm(span=EMA_LEN, adjust=False, min_periods=EMA_LEN).mean()
    lo14 = low.rolling(14, min_periods=14).min()
    hi14 = high.rolling(14, min_periods=14).max()
    stoch = 100.0 * (close - lo14) / (hi14 - lo14).replace(0.0, np.nan)
    df["stk"] = stoch.rolling(3, min_periods=3).mean()
    df["val_avg"] = (close * vol).rolling(20, min_periods=20).mean()
    df["lo5"] = low.rolling(5, min_periods=5).min()
    df["stk_min"] = df["stk"].rolling(5, min_periods=5).min()
    df["rsi_min"] = pd.to_numeric(df["rsi14"], errors="coerce").rolling(5, min_periods=5).min()
    h_r = high.rolling(RANGE_LEN, min_periods=RANGE_LEN).max()
    l_r = low.rolling(RANGE_LEN, min_periods=RANGE_LEN).min()
    df["rpos"] = (close - l_r) / np.maximum(h_r - l_r, 1.0)   # IDX mintick is Rp1
    df["str_e"] = (df["ema"] - close) / df["atr"]

    ph = _pivot_series(high.to_numpy(float), PIVOT, PIVOT, True)
    pl = _pivot_series(low.to_numpy(float), PIVOT, PIVOT, False)
    c = close.to_numpy(float)
    regime = np.full(len(df), SIDE)
    last_ph = prev_ph = last_pl = prev_pl = np.nan
    for i in range(len(df)):
        if not np.isnan(ph[i]):
            prev_ph, last_ph = last_ph, ph[i]
        if not np.isnan(pl[i]):
            prev_pl, last_pl = last_pl, pl[i]
        if not (np.isnan(prev_ph) or np.isnan(prev_pl)):
            hh, hl = last_ph > prev_ph, last_pl > prev_pl
            lh, ll = last_ph < prev_ph, last_pl < prev_pl
            regime[i] = UP if hh and hl else DOWN if lh and ll else SIDE
            if c[i] > last_ph:
                regime[i] = UP
            elif c[i] < last_pl:
                regime[i] = DOWN
    df["regime"] = regime
    return df


def _setup(df: pd.DataFrame) -> np.ndarray:
    """The Pine's ``setup and liquid and warm``, per bar."""
    c = pd.to_numeric(df["close"], errors="coerce")
    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    deep = pd.Series(False, index=df.index)
    for reg, (pos, gap, rsi, stk) in DEPTH.items():
        deep |= ((df["regime"] == reg) & (df["rpos"] <= pos) & (df["str_e"] >= gap)
                 & (df["rsi_min"] <= rsi) & (df["stk_min"] <= stk))
    trig = (c > h.shift(1)) & (c > o)
    near = (c - df["lo5"]) <= NEAR_LOW_ATR * df["atr"]
    liquid = df["val_avg"] >= MIN_TURNOVER
    warm = (df.index >= RANGE_LEN) & df["atr"].notna()
    return (deep & trig & near & liquid & warm).to_numpy(bool)


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Signals at the close, fills at the next open, stop resting in between."""
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    stk = df["stk"].to_numpy(float)
    lo5 = df["lo5"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    regime = df["regime"].to_numpy(int)
    setup = _setup(df)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n

    in_pos = False
    pending_buy = pending_sell = False
    entry_price = sl = np.nan
    entry_date = None
    entry_idx = -1
    entry_code = ""
    last_exit = -1000
    buy_code = ""
    buy_sl = np.nan

    def close_trade(i: int, code: str, price: float) -> None:
        codes[i] = code if not codes[i] else f"{codes[i]} + {code}"
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": entry_code, "exit_date": dates[i],
            "exit_price": price, "exit_code": code,
            "bars_held": i - entry_idx,
            "gross_return_pct": (price / entry_price - 1.0) * 100.0,
            "resolved": True,
        })

    for i in range(n):
        # 1. Orders from the previous close fill at this open.
        if pending_sell:
            pending_sell, in_pos = False, False
            close_trade(i, "TP Swing High", o[i])
            last_exit = i
        if pending_buy:
            pending_buy, in_pos = False, True
            entry_price, entry_date, entry_idx, entry_code = o[i], dates[i], i, buy_code
            sl = buy_sl
            codes[i] = entry_code if not codes[i] else f"{codes[i]} + {entry_code}"
        # 2. The resting stop, intraday from the fill bar on.
        if in_pos and lo[i] <= sl:
            in_pos = False
            close_trade(i, "CL Stop loss", min(o[i], sl))
            last_exit = i
        # 3. At the close: a sell for tomorrow, or a buy for tomorrow.
        if in_pos and stk[i] >= X_STOCH and c[i] < o[i] and c[i] > entry_price * (1 + MIN_PROFIT / 100):
            pending_sell = True
        elif not in_pos and setup[i] and i - last_exit > COOL:
            pending_buy = True
            buy_code = f"BUY {REGIME_NAME[int(regime[i])]}"
            buy_sl = floor_tick(lo5[i] - SL_ATR * atr[i])
        position_arr[i] = int(in_pos)

    if in_pos:
        last = n - 1
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": entry_code, "exit_date": dates[last],
            "exit_price": c[last], "exit_code": "OPEN",
            "bars_held": last - entry_idx,
            "gross_return_pct": (c[last] / entry_price - 1.0) * 100.0,
            "resolved": False,
        })

    # Armed with nothing bought: a buy signalled at the last close, to be
    # filled at tomorrow's open.
    setup_arr = [False] * n
    if n and pending_buy:
        setup_arr[-1] = True
    return codes, trades, {"position": position_arr, "setup": setup_arr}


if __name__ == "__main__":
    assert floor_tick(199.7) == 199.0 and floor_tick(487) == 486.0 and floor_tick(1234) == 1230.0
    assert floor_tick(4999) == 4990.0 and floor_tick(5010) == 5000.0

    # Execution on a hand-built frame (indicators and regime given directly,
    # setup forced on chosen bars): a signal at bar 1's close fills at bar 2's
    # open; a swing-high sell at bar 4's close fills at bar 5's open; the next
    # signal waits out the 2-bar cool-down; a stop is taken at its level.
    n = 14
    f = pd.DataFrame({
        "open": [1000.0] * n, "high": [1010.0] * n, "low": [990.0] * n, "close": [1005.0] * n,
        "stk": [50.0] * n, "lo5": [980.0] * n, "atr": [10.0] * n, "regime": [UP] * n,
    })
    f["date"] = pd.date_range("2024-01-01", periods=n, freq="D")
    forced = np.zeros(n, bool)
    forced[[1, 5, 6, 8]] = True
    f.loc[2, "open"] = 1002.0                                 # fill price
    f.loc[4, ["stk", "open", "close"]] = [85.0, 1030.0, 1020.0]   # red, +1.8% over 1002
    f.loc[5, "open"] = 1015.0                                 # sell fills here
    f.loc[10, "low"] = 940.0                                  # stop 980-25 = 955 -> 955
    import sys
    this = sys.modules[__name__]
    real_setup, this._setup = _setup, (lambda _df: forced)
    codes, trades, lines = run_state_machine(f)
    this._setup = real_setup
    fired = [(i, x) for i, x in enumerate(codes) if x]
    # bar 5 sells; bars 5 and 6 are inside the cool-down; bar 8 signals, fills at 9
    assert fired == [(2, "BUY UPTREND"), (5, "TP Swing High"), (9, "BUY UPTREND"),
                     (10, "CL Stop loss")], fired
    assert trades[0]["entry_price"] == 1002.0 and trades[0]["exit_price"] == 1015.0, trades[0]
    assert trades[1]["exit_price"] == 955.0, trades[1]
    assert not any(lines["setup"])
    print("swing_adaptive self-check ok:", fired)
