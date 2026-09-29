"""Uptrend Buy The Dip strategy (MA20 > MA50, EOD).

A port of the Pine Script "Uptrend Buy The Dip - MA20>MA50" at its defaults
(SMA, dip = pullback to MA20, bounce candle required), in the same shape as
``market_structure`` so ``trade_log`` serves it unchanged.

    BUY Golden cross   SMA20 crosses over SMA50
    BUY Dip            while SMA20 > SMA50: the low touches SMA20, the close
                       stays over SMA50, on a green candle
                       Up to 5 entries stacked (pyramiding), at least 5 bars
                       apart; none on an exit bar.
    Exit (all)         a close under the trailing stop (close minus 3 ATR(14),
                       only rising, reset when flat), or SMA20 crossing under
                       SMA50

Each entry is its own row, as the Pine lists them; all open entries leave
together. The Pine has no take-profit, so each row's exit counts as TP when
it fills above that row's buy price, otherwise CL.

The Pine fills at the signal close (``process_orders_on_close = true``);
with end-of-day data an order can only go in for the next session, so here
every BUY and the exit fill at the next open, as for A and B. A BUY
signalled at the last close shows as a WATCHLIST row when nothing is held.
The Pine's 2020 start date, sizing and 0.2% commission are not ported; the
page's fee model (0.15% buy, 0.25% sell) applies instead.

A *monitoring* screener with no validated edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idxcore.compute.market_structure import _all_bars, _ticker_meta  # noqa: F401
from idxcore.compute.reversal_sniper import _rma

FAST, SLOW = 20, 50
ATR_MULT = 3.0
MAX_ENTRIES = 5
MIN_GAP = 5

MIN_BARS = SLOW + 1

CATEGORY = {"BUY": "BUY", "TP": "TAKE PROFIT", "CL": "CUT LOSS"}


def category_of(code: str) -> str:
    return CATEGORY.get(code.split(" ", 1)[0], "OTHER")


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    """SMA20/SMA50, ATR(14) and the dip candle the Pine reads."""
    df = history.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    opn = pd.to_numeric(df["open"], errors="coerce")

    fast = close.rolling(FAST, min_periods=FAST).mean()
    slow = close.rolling(SLOW, min_periods=SLOW).mean()
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    df["atr"] = _rma(tr, 14)
    df["uptrend"] = fast > slow
    df["golden"] = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    df["death"] = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    df["dip"] = (low <= fast) & (close > slow) & (close > opn)
    return df


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Signals at the close, fills at the next open, all entries exit together."""
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    atr = df["atr"].to_numpy(float)
    uptrend, golden = df["uptrend"].to_numpy(bool), df["golden"].to_numpy(bool)
    death, dip = df["death"].to_numpy(bool), df["dip"].to_numpy(bool)
    # Optional extra gate on entries (buy_the_dip_filtered sets it).
    entry_ok = df["entry_ok"].to_numpy(bool) if "entry_ok" in df else np.ones(len(df), bool)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n

    held: list[dict] = []          # open entries
    pending_buy = ""               # entry code to fill at the next open
    pending_exit = ""              # exit reason to fill at the next open
    trail = np.nan
    last_entry = -10**9            # bar of the last BUY signal

    def mark(i: int, code: str) -> None:
        codes[i] = code if not codes[i] else f"{codes[i]} + {code}"

    for i in range(n):
        # 1. Orders from the previous close fill at this open.
        if pending_exit:
            for e in held:
                code = ("TP " if o[i] > e["entry_price"] else "CL ") + pending_exit
                trades.append({**e, "exit_date": dates[i], "exit_price": o[i], "exit_code": code,
                               "bars_held": i - e["idx"],
                               "gross_return_pct": (o[i] / e["entry_price"] - 1.0) * 100.0,
                               "resolved": True})
            mark(i, pending_exit)
            held, pending_exit = [], ""
        if pending_buy:
            held.append({"entry_date": dates[i], "entry_price": o[i], "entry_code": pending_buy,
                         "idx": i})
            mark(i, pending_buy)
            pending_buy = ""
        # 2. At the close: exit first, then a BUY, then the trailing stop.
        in_pos = bool(held)
        exit_trail = in_pos and not np.isnan(trail) and c[i] < trail
        exit_trend = in_pos and death[i]
        exit_now = exit_trail or exit_trend
        if exit_now:
            pending_exit = "MA cross down" if exit_trend else "Trailing stop"
        buy = (uptrend[i] and entry_ok[i] and len(held) < MAX_ENTRIES and i - last_entry >= MIN_GAP
               and not exit_now and (golden[i] or dip[i]))
        if buy:
            pending_buy = "BUY Golden cross" if golden[i] else "BUY Dip"
            last_entry = i
        new = c[i] - atr[i] * ATR_MULT
        if (in_pos and not exit_now) or buy:
            trail = new if np.isnan(trail) else max(trail, new)
        else:
            trail = np.nan
        position_arr[i] = int(in_pos)

    last = n - 1
    for e in held:
        trades.append({**e, "exit_date": dates[last], "exit_price": c[last], "exit_code": "OPEN",
                       "bars_held": last - e["idx"],
                       "gross_return_pct": (c[last] / e["entry_price"] - 1.0) * 100.0,
                       "resolved": False})
    for t in trades:
        t.pop("idx", None)
    trades.sort(key=lambda t: t["entry_date"])

    # A buy signalled at the last close, to be filled at tomorrow's open.
    setup_arr = [False] * n
    if n and pending_buy:
        setup_arr[-1] = True
    return codes, trades, {"position": position_arr, "setup": setup_arr}


if __name__ == "__main__":
    # Golden cross at bar 1 (fill 2), a dip at bar 7 (fill 8), a close under
    # the trail at bar 10 -> both entries sold at bar 11's open. The dip at
    # bar 4 is refused: under 5 bars after the bar-1 signal.
    n = 14
    f = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": [100.0] * n, "close": [101.0] * n, "atr": [1.0] * n,
        "uptrend": [True] * n, "golden": [False] * n, "death": [False] * n, "dip": [False] * n,
    })
    f.loc[1, "golden"] = True
    f.loc[[4, 7], "dip"] = True
    f.loc[2, "open"], f.loc[8, "open"] = 100.0, 104.0
    f.loc[5:9, "close"] = 106.0               # trail climbs to 103
    f.loc[10, "close"] = 102.0                # under 103 -> exit at bar 11
    f.loc[11, "open"] = 102.0
    codes, trades, lines = run_state_machine(f)
    got = [(str(t["entry_code"]), t["entry_price"], t["exit_code"], t["exit_price"]) for t in trades]
    assert got == [("BUY Golden cross", 100.0, "TP Trailing stop", 102.0),
                   ("BUY Dip", 104.0, "CL Trailing stop", 102.0)], got
    assert category_of(trades[1]["exit_code"]) == "CUT LOSS"
    assert not any(lines["setup"])
    print("buy_the_dip self-check ok:", [(i, x) for i, x in enumerate(codes) if x])
