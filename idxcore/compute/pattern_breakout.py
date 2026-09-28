"""Pattern Breakout strategy: VCP and Double Bottom (EOD).

A port of the Pine Script "Pattern Breakout: VCP & Double Bottom" at its
defaults (both patterns), in the same shape as ``market_structure`` so
``trade_log`` serves it unchanged. Pivots are 5/5.

    BUY VCP        a base that opens on a swing high, then 2+ pullbacks each
                   <= 0.85x the one before (first <= 35%, last <= 10%), a base
                   20-260 bars long, the Minervini trend template (close >
                   SMA50 > SMA150 > SMA200, SMA200 rising over 20 bars, >= 25%
                   over the 52-week low, <= 25% under the 52-week high), dry
                   volume the bar before (SMA10 < 0.85x SMA50), then a close
                   through the last pullback's high, at most 5% over it
    BUY DB         two swing lows within 4% of each other, 15-120 bars apart,
                   a neckline 10-40% over them; a close through the neckline
                   (at most 5% over it) within 40 bars, before a close more
                   than 4% under the lows
                   Both need volume >= 1.5x its 50-bar average and a 20-bar
                   average turnover >= Rp5bn.
    Stop           the pattern low - 0.5%, never more than 8% under the close
    TP 2R          half the position at entry + 2x risk; the rest's stop then
                   moves to the entry and it is sold at the next open after a
                   close under EMA21
    CL Stop loss   the stop, intraday

Execution follows the Pine with ``process_orders_on_close = false``: the BUY
fills at the next open; stop and limit are resting orders filled at their
level, or at the open on a gap through. When one bar reaches both, the stop is
taken (never resolve a same-bar tie in the trade's favour). The two halves are
one row: its exit price is their average and its exit date the last fill. A
trade whose second half has not been sold yet stays OPEN.

A signal on the last bar has not been filled yet: it shows as a WATCHLIST row.
The Pine's position sizing (1% risk), slippage and labels are not ported; the
page's fee model (0.15% buy, 0.25% sell) applies instead.

A *monitoring* screener with no validated edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idxcore.compute.market_structure import _all_bars, _pivot_series, _ticker_meta  # noqa: F401

PIVOT = 5
# VCP
MIN_CONTR, MAX_FIRST, MAX_LAST, CONTR_RATIO = 2, 35.0, 10.0, 0.85
MIN_BASE, MAX_BASE = 20, 260
DRY_UP = 0.85
MIN_FROM_LOW, MAX_FROM_HIGH = 25.0, 25.0
# Double Bottom
DB_TOL, DB_MIN_GAP, DB_MAX_GAP = 4.0, 15, 120
DB_MIN_DEPTH, DB_MAX_DEPTH, DB_EXPIRE = 10.0, 40.0, 40
# Entry and exit
VOL_MULT = 1.5
MAX_EXT = 5.0               # % over the pivot a breakout may close
MIN_TURNOVER = 5e9          # Rp5bn average daily value
MAX_STOP = 8.0              # %
TP1_R = 2.0
TP1_SHARE = 0.5
TRAIL_LEN = 21

#: A TP row blends the 2R half with the rest, sold at its breakeven stop or at
#: the next open after the trail; a gap down can pull the blend under the
#: entry, so trade_log's "a TP is a profit" check skips this strategy.
TP_FILLS_NEXT_OPEN = True
#: The 2R half can sell on the BUY bar itself, above every later high; the
#: row's H+1 peak (Max % FL) can then sit under its P&L (ZBRA, 2021-08-26).
PARTIAL_EXIT_ON_ENTRY_BAR = True

MIN_BARS = 60  # the Double Bottom needs no long-term averages; VCP waits for its own

CATEGORY = {"BUY": "BUY", "TP": "TAKE PROFIT", "CL": "CUT LOSS"}


def category_of(code: str) -> str:
    return CATEGORY.get(code.split(" ", 1)[0], "OTHER")


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    """Every average the Pine reads."""
    df = history.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")

    sma50 = close.rolling(50, min_periods=50).mean()
    sma150 = close.rolling(150, min_periods=150).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    hi52 = high.rolling(252, min_periods=252).max()
    lo52 = low.rolling(252, min_periods=252).min()
    df["trend_ok"] = ((close > sma50) & (sma50 > sma150) & (sma150 > sma200)
                      & (sma200 > sma200.shift(20))
                      & (close >= lo52 * (1 + MIN_FROM_LOW / 100))
                      & (close >= hi52 * (1 - MAX_FROM_HIGH / 100)))
    df["liq_ok"] = (close * vol).rolling(20, min_periods=20).mean() >= MIN_TURNOVER
    vol50 = vol.rolling(50, min_periods=50).mean()
    vol10 = vol.rolling(10, min_periods=10).mean()
    df["vol_ok"] = vol >= vol50 * VOL_MULT
    df["dry_ok"] = (vol10 < vol50 * DRY_UP).shift(1, fill_value=False)
    df["trail_ma"] = close.ewm(span=TRAIL_LEN, adjust=False, min_periods=TRAIL_LEN).mean()
    df["ph"] = _pivot_series(high.to_numpy(float), PIVOT, PIVOT, True)
    df["pl"] = _pivot_series(low.to_numpy(float), PIVOT, PIVOT, False)
    return df


def _signals(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per bar: the pattern's stop base (VCP or DB low), or NaN; and its code.

    The pattern trackers run on every bar, in or out of a position, as in the
    Pine; whether a signal is taken is the state machine's business.
    """
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    ph, pl = df["ph"].to_numpy(float), df["pl"].to_numpy(float)
    trend_ok, liq_ok = df["trend_ok"].to_numpy(bool), df["liq_ok"].to_numpy(bool)
    vol_ok, dry_ok = df["vol_ok"].to_numpy(bool), df["dry_ok"].to_numpy(bool)

    n = len(df)
    fire = np.zeros(n, bool)
    base_low = np.full(n, np.nan)
    code = [""] * n

    base_high = last_ph = vcp_pivot = vcp_low = np.nan
    base_start = -1
    depths: list[float] = []
    last_pl = neck_cand = db_neck = db_low = np.nan
    last_pl_bar = db_armed = -1

    for i in range(n):
        p_bar = i - PIVOT
        # --- VCP tracker
        if not np.isnan(ph[i]):
            if np.isnan(base_high) or ph[i] > base_high:
                base_high, base_start = ph[i], p_bar
                depths.clear()
                vcp_pivot = vcp_low = np.nan
                last_ph = ph[i]
            else:
                last_ph = ph[i] if np.isnan(last_ph) else max(last_ph, ph[i])
        if not np.isnan(pl[i]) and not np.isnan(base_high):
            if not np.isnan(last_ph):
                depths.append((last_ph - pl[i]) / last_ph * 100)
                vcp_pivot, vcp_low, last_ph = last_ph, pl[i], np.nan
                if len(depths) > 6:
                    depths.pop(0)
            elif depths and not np.isnan(vcp_pivot) and pl[i] < vcp_low:
                # a lower low with no swing high between: the last pullback was deeper
                depths[-1] = (vcp_pivot - pl[i]) / vcp_pivot * 100
                vcp_low = pl[i]
        streak, first_d, last_d = 0, np.nan, np.nan
        if depths:
            streak, last_d = 1, depths[-1]
            first_d = last_d
            for k in range(len(depths) - 1, 0, -1):
                if depths[k] <= depths[k - 1] * CONTR_RATIO:
                    streak += 1
                    first_d = depths[k - 1]
                else:
                    break
        base_len = 0 if base_start < 0 else i - base_start
        vcp_setup = (not np.isnan(vcp_pivot) and streak >= MIN_CONTR and first_d <= MAX_FIRST
                     and last_d <= MAX_LAST and MIN_BASE <= base_len <= MAX_BASE
                     and trend_ok[i] and liq_ok[i])
        go_vcp = (vcp_setup and dry_ok[i] and vol_ok[i]
                  and vcp_pivot < c[i] <= vcp_pivot * (1 + MAX_EXT / 100))

        # --- Double Bottom tracker
        if not np.isnan(ph[i]) and not np.isnan(last_pl) and p_bar > last_pl_bar:
            neck_cand = ph[i] if np.isnan(neck_cand) else max(neck_cand, ph[i])
        if not np.isnan(pl[i]):
            if not np.isnan(last_pl) and not np.isnan(neck_cand):
                gap = p_bar - last_pl_bar
                diff = (pl[i] - last_pl) / last_pl * 100
                lowest2 = min(pl[i], last_pl)
                depth = (neck_cand - lowest2) / lowest2 * 100
                if (DB_MIN_GAP <= gap <= DB_MAX_GAP and abs(diff) <= DB_TOL
                        and DB_MIN_DEPTH <= depth <= DB_MAX_DEPTH):
                    db_neck, db_low, db_armed = neck_cand, lowest2, i
            last_pl, last_pl_bar, neck_cand = pl[i], p_bar, np.nan
        if not np.isnan(db_neck) and (c[i] < db_low * (1 - DB_TOL / 100) or i - db_armed > DB_EXPIRE):
            db_neck = np.nan
        go_db = (not go_vcp and not np.isnan(db_neck) and vol_ok[i] and liq_ok[i]
                 and db_neck < c[i] <= db_neck * (1 + MAX_EXT / 100))

        if go_vcp:
            fire[i], base_low[i] = True, vcp_low
            code[i] = f"BUY VCP {streak}T"
        elif go_db:
            fire[i], base_low[i] = True, db_low
            code[i] = "BUY Double Bottom"

        # a used setup, or one the price has run away from, is spent
        if go_vcp or (not np.isnan(vcp_pivot) and c[i] > vcp_pivot * (1 + MAX_EXT / 100)):
            vcp_pivot = np.nan
        if go_db or (not np.isnan(db_neck) and c[i] > db_neck * (1 + MAX_EXT / 100)):
            db_neck = np.nan

    return fire, base_low, code


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Signals at the close, fills at the next open, stop and 2R limit resting."""
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    trail_ma = df["trail_ma"].to_numpy(float)
    fire, base_low, sig_code = _signals(df)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n

    in_pos = tp1_done = False
    pending_buy = pending_sell = False
    entry_price = stop = limit = tp1_price = np.nan
    entry_date = None
    entry_idx = -1
    entry_code = buy_code = ""
    buy_stop = buy_limit = np.nan

    def mark(i: int, code: str) -> None:
        codes[i] = code if not codes[i] else f"{codes[i]} + {code}"

    def close_trade(i: int, code: str, price: float) -> None:
        # One row per trade: after the 2R half, the price is the two halves' average.
        px = TP1_SHARE * tp1_price + (1 - TP1_SHARE) * price if tp1_done else price
        mark(i, code)
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": entry_code, "exit_date": dates[i],
            "exit_price": px, "exit_code": code,
            "bars_held": i - entry_idx,
            "gross_return_pct": (px / entry_price - 1.0) * 100.0,
            "resolved": True,
        })

    for i in range(n):
        # 1. Orders from the previous close fill at this open.
        if pending_sell:
            pending_sell, in_pos = False, False
            close_trade(i, "TP 2R + Trail EMA21", o[i])
        if pending_buy:
            pending_buy, in_pos, tp1_done = False, True, False
            entry_price, entry_date, entry_idx, entry_code = o[i], dates[i], i, buy_code
            stop, limit = buy_stop, buy_limit
            mark(i, entry_code)
        # 2. Resting orders, intraday. The stop first: a bar that reaches both
        #    is resolved against the trade.
        if in_pos and lo[i] <= stop:
            in_pos = False
            close_trade(i, "TP 2R + BE stop" if tp1_done else "CL Stop loss", min(o[i], stop))
        elif in_pos and not tp1_done and h[i] >= limit:
            tp1_done, tp1_price = True, max(o[i], limit)
        # 3. At the close.
        if in_pos:
            if i == entry_idx:
                # From the fill on, the 2R target is measured from the price paid.
                limit = entry_price + TP1_R * (entry_price - stop)
            if tp1_done:
                stop = max(stop, entry_price)
                if c[i] < trail_ma[i]:
                    pending_sell = True
        elif fire[i]:
            stp = max(base_low[i] * 0.995, c[i] * (1 - MAX_STOP / 100))
            if stp < c[i]:
                pending_buy, buy_code = True, sig_code[i]
                buy_stop, buy_limit = stp, c[i] + TP1_R * (c[i] - stp)
        position_arr[i] = int(in_pos)

    if in_pos:
        last = n - 1
        px = TP1_SHARE * tp1_price + (1 - TP1_SHARE) * c[last] if tp1_done else c[last]
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": entry_code, "exit_date": dates[last],
            "exit_price": px, "exit_code": "OPEN",
            "bars_held": last - entry_idx,
            "gross_return_pct": (px / entry_price - 1.0) * 100.0,
            "resolved": False,
        })

    # A buy signalled at the last close, to be filled at tomorrow's open.
    setup_arr = [False] * n
    if n and pending_buy:
        setup_arr[-1] = True
    return codes, trades, {"position": position_arr, "setup": setup_arr}


if __name__ == "__main__":
    import sys

    # Execution on a hand-built frame (signals forced on chosen bars):
    # signal at bar 1 (close 1000, stop 950 -> limit 1100), fill at bar 2's
    # open 1010 (limit re-based to 1010 + 2x60 = 1130); bar 4 reaches 1130,
    # half sold; bar 5 closes under EMA21, the rest sold at bar 6's open.
    # Then a signal at bar 7, filled at 8, stopped at bar 9.
    n = 12
    f = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": [1000.0] * n, "high": [1010.0] * n, "low": [990.0] * n, "close": [1000.0] * n,
        "trail_ma": [900.0] * n,
    })
    f.loc[2, "open"] = 1010.0
    f.loc[4, "high"] = 1140.0
    f.loc[5, ["open", "high", "low", "close", "trail_ma"]] = [1100.0, 1100.0, 1050.0, 1080.0, 1090.0]
    f.loc[6, "open"] = 1070.0
    f.loc[9, "low"] = 900.0
    fire = np.zeros(n, bool)
    fire[[1, 7]] = True
    this = sys.modules[__name__]
    real = this._signals
    this._signals = lambda _df: (fire, np.full(n, 950 / 0.995), ["BUY VCP 2T"] * n)
    codes, trades, lines = run_state_machine(f)
    this._signals = real
    assert [(t["exit_code"], round(t["exit_price"], 2)) for t in trades] == [
        ("TP 2R + Trail EMA21", 1100.0), ("CL Stop loss", 950.0)], trades
    assert trades[0]["entry_price"] == 1010.0 and trades[1]["entry_price"] == 1000.0, trades
    assert category_of(trades[0]["exit_code"]) == "TAKE PROFIT"
    assert not any(lines["setup"])
    print("pattern_breakout self-check ok:", [(i, x) for i, x in enumerate(codes) if x])
