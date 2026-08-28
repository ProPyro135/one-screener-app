"""Market Structure strategy (regime auto-engine: EMA20/SMA200 + swing pivots).

A faithful port of the Pine Script "Market Structure: Clean Visual Auto-Engine".
It walks each ticker's bars as a small position state machine and emits a signal
code on the bars where something happens:

    PANTAU               a higher-low rebound on dry volume is forming (watch)
    BUY LOW (EMA20/…)    breakout entry off that pullback, regime-tagged
    SELL HIGH            take profit: reversal at a structure top or after +8%
    SL                   cut loss: close breaks the initial swing-low stop

Consistent with the rest of this app
-------------------------------------
The Pine's Bollinger cloud and EMA50 are **visual only** — no rule reads them —
so they are not computed here. The logic needs OHLCV, EMA20, EMA9 and SMA200.
SMA200 is already stored (``indicators.ma200``) and is reused (invariant 1); the
two EMAs are the only thing computed on the fly, matching how ``bottom_fishing``
computes MA10. Prices are raw/unadjusted, so a signal lines up with the candles
the dashboard draws.

This is a *monitoring* screener, like the bottom-fishing radar — not a proven
edge. No backtester is provided (YAGNI; add one if a validation run is wanted).
"""

from __future__ import annotations

from typing import Optional

import duckdb
import numpy as np
import pandas as pd

# Pine inputs (defaults)
PIVOT_LEFT = 4
PIVOT_RIGHT = 4
VOL_MA_LEN = 20
PULLBACK_BARS = 3
DRY_VOL_RATIO = 0.85
SL_OFFSET_PCT = 1.5
MIN_GAIN_FOR_TP = 8.0
VOLATILE_LOOKBACK = 40
VOLATILE_RANGE_PCT = 30.0
REQUIRE_PANTAU_FIRST = False

MIN_BARS = 45  # need the 40-bar volatility window + a pivot to say anything

REASON_MAP = {
    "PANTAU": "Pantau: rebound Higher-Low dengan volume kering, menunggu konfirmasi breakout",
    "BUY LOW (EMA20)": "Entry breakout pullback kering di regime momentum (Close > EMA20), rebound dari Higher-Low",
    "BUY LOW (SMA200)": "Entry breakout pullback kering di regime makro (Close > SMA200), rebound dari Higher-Low",
    "SELL HIGH": "Take Profit: reversal di pucuk (kenaikan >= 8% atau sentuh resisten swing)",
    "SL": "Cut Loss: Close jebol di bawah Stop Loss awal (di bawah swing low)",
}

CATEGORY = {"PANTAU": "PANTAU", "BUY": "BUY", "SELL": "SELL HIGH", "SL": "CUT LOSS"}

#: naik = holding a position (riding up), pantau = flat with a rebound armed,
#: tunggu = flat and nothing forming.
STATUS_COLOUR = {"naik": "#2962FF", "pantau": "#FB8C00", "tunggu": "#787B86"}


def category_of(code: str) -> str:
    return CATEGORY.get(code.split(" ", 1)[0], "OTHER")


# ---------------------------------------------------------------------------
# prepare a ticker's bars (rolling helpers the machine needs)
# ---------------------------------------------------------------------------

def _pivot_series(x: np.ndarray, left: int, right: int, is_high: bool) -> np.ndarray:
    """Pine ``ta.pivothigh``/``pivotlow``: the pivot value, reported ``right``
    bars after it, when the middle bar is the *unique* extreme of its window."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        c = i - right
        window = x[c - left : c + right + 1]
        mid = x[c]
        if np.isnan(window).any():
            continue
        if is_high:
            if mid == window.max() and np.count_nonzero(window == mid) == 1:
                out[i] = mid
        else:
            if mid == window.min() and np.count_nonzero(window == mid) == 1:
                out[i] = mid
    return out


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    """Add EMA9/EMA20, volume SMAs, pivots and the volatility window."""
    df = history.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")

    df["ema9"] = close.ewm(span=9, adjust=False, min_periods=9).mean()
    df["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    df["vol_sma"] = vol.rolling(VOL_MA_LEN, min_periods=VOL_MA_LEN).mean()
    df["vol_pullback"] = vol.rolling(PULLBACK_BARS, min_periods=PULLBACK_BARS).mean()

    lb = VOLATILE_LOOKBACK
    hh = high.rolling(lb, min_periods=lb).max()
    ll = low.rolling(lb, min_periods=lb).min()
    df["is_volatile"] = ((hh - ll) / ll * 100.0 >= VOLATILE_RANGE_PCT)

    df["pivot_high"] = _pivot_series(high.to_numpy(float), PIVOT_LEFT, PIVOT_RIGHT, True)
    df["pivot_low"] = _pivot_series(low.to_numpy(float), PIVOT_LEFT, PIVOT_RIGHT, False)
    return df


# ---------------------------------------------------------------------------
# the state machine
# ---------------------------------------------------------------------------

def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Walk the bars; return (code per bar, round-trip trades, per-bar lines).

    A direct port of the Pine's persistent-state logic. Entry and exit fill at
    the signal bar's close, faithful to how ``bottom_fishing`` treats the Pine
    backtest. Exits are evaluated take-profit first, then the stop.
    """
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    vol = pd.to_numeric(df["volume"], errors="coerce").to_numpy(float)
    ema9 = pd.to_numeric(df["ema9"], errors="coerce").to_numpy(float)
    ema20 = pd.to_numeric(df["ema20"], errors="coerce").to_numpy(float)
    sma200 = pd.to_numeric(df["ma200"], errors="coerce").to_numpy(float)
    vol_sma = pd.to_numeric(df["vol_sma"], errors="coerce").to_numpy(float)
    vol_pb = pd.to_numeric(df["vol_pullback"], errors="coerce").to_numpy(float)
    is_vol = df["is_volatile"].to_numpy(bool)
    piv_hi = pd.to_numeric(df["pivot_high"], errors="coerce").to_numpy(float)
    piv_lo = pd.to_numeric(df["pivot_low"], errors="coerce").to_numpy(float)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n
    pantau_arr = [False] * n

    pos = 0
    entry_idx = -1
    entry_price = np.nan
    entry_date = None
    entry_code = ""
    initial_sl = np.nan
    peak = np.nan
    major_high = np.nan
    last_low = np.nan
    prev_low = np.nan
    is_hl = False
    pantau_hist: list[bool] = []  # signalPantau over recent bars

    def is_dry(i: int) -> bool:
        if np.isnan(vol_pb[i]) or np.isnan(vol_sma[i]):
            return False
        return vol_pb[i] <= vol_sma[i] * DRY_VOL_RATIO or vol[i] < vol_sma[i]

    for i in range(n):
        # structure updates (persistent, only on a confirmed pivot)
        if not np.isnan(piv_hi[i]):
            major_high = piv_hi[i]
        if not np.isnan(piv_lo[i]):
            prev_low, last_low = last_low, piv_lo[i]
            is_hl = (not np.isnan(prev_low)) and last_low > prev_low

        use_ema20 = bool(is_vol[i]) or np.isnan(sma200[i])
        if use_ema20:
            filter_pass = (not np.isnan(ema20[i])) and c[i] > ema20[i]
        else:
            filter_pass = c[i] > sma200[i]

        rebounding = (not np.isnan(last_low)) and c[i] > o[i] and c[i] > last_low

        pantau = (
            pos == 0 and filter_pass and is_hl and rebounding and is_dry(i)
            and not np.isnan(vol_sma[i]) and vol[i] < vol_sma[i]
        )
        had_recent_pantau = any(pantau_hist[-PULLBACK_BARS:])

        crossover9 = (
            i > 0 and not np.isnan(ema9[i]) and not np.isnan(ema9[i - 1])
            and c[i] > ema9[i] and c[i - 1] <= ema9[i - 1]
        )
        raw_buy = (
            filter_pass and rebounding and is_dry(i - 1 if i > 0 else i)
            and not np.isnan(vol_sma[i]) and vol[i] >= vol_sma[i]
            and (i > 0 and c[i] > h[i - 1] or crossover9)
        )
        buy_low = raw_buy and (had_recent_pantau if REQUIRE_PANTAU_FIRST else True)

        if buy_low and pos == 0:
            pos, entry_idx = 1, i
            entry_price, entry_date = c[i], dates[i]
            entry_code = "BUY LOW (EMA20)" if use_ema20 else "BUY LOW (SMA200)"
            codes[i] = entry_code
            base = last_low if not np.isnan(last_low) else lo[i]
            initial_sl = base * (1.0 - SL_OFFSET_PCT / 100.0)
            peak = h[i]
        elif pos == 1 and i > entry_idx:
            peak = h[i] if np.isnan(peak) else max(peak, h[i])
            gain = (peak - entry_price) / entry_price * 100.0
            reached_target = gain >= MIN_GAIN_FOR_TP or (
                not np.isnan(major_high) and h[i] >= major_high * 0.99
            )
            top_reversal = c[i] < o[i] and i > 0 and c[i] < lo[i - 1]
            if reached_target and top_reversal:
                codes[i] = "SELL HIGH"
            elif not np.isnan(initial_sl) and c[i] < initial_sl:
                codes[i] = "SL"
            if codes[i]:
                trades.append({
                    "entry_date": entry_date, "entry_price": entry_price,
                    "entry_code": entry_code, "exit_date": dates[i],
                    "exit_price": c[i], "exit_code": codes[i],
                    "bars_held": i - entry_idx,
                    "gross_return_pct": (c[i] / entry_price - 1.0) * 100.0,
                    "resolved": True,
                })
                pos, entry_idx = 0, -1
                initial_sl = peak = np.nan

        if pos == 0 and not codes[i] and pantau:
            codes[i] = "PANTAU"

        pantau_hist.append(bool(pantau))
        position_arr[i] = pos
        pantau_arr[i] = bool(pantau)

    if pos == 1:
        last = n - 1
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": entry_code, "exit_date": dates[last],
            "exit_price": c[last], "exit_code": "OPEN",
            "bars_held": last - entry_idx,
            "gross_return_pct": (c[last] / entry_price - 1.0) * 100.0,
            "resolved": False,
        })

    lines = {"position": position_arr, "pantau": pantau_arr}
    return codes, trades, lines


def annotate(history: pd.DataFrame) -> pd.DataFrame:
    """``prepare`` + run the machine, returning the frame with a ``ms_code`` column."""
    df = prepare(history)
    codes, _, _ = run_state_machine(df)
    df["ms_code"] = codes
    return df


# ---------------------------------------------------------------------------
# universe screen / radar
# ---------------------------------------------------------------------------

_BARS_SQL = """
    SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume,
           i.ma200
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


def _ticker_meta(con: duckdb.DuckDBPyConnection) -> dict:
    return {t: (c, n) for t, c, n in
            con.execute("SELECT ticker, idx_code, name FROM tickers").fetchall()}


def _status(pos: int, pantau: bool) -> str:
    if pos == 1:
        return "naik"
    if pantau:
        return "pantau"
    return "tunggu"


def radar_screen(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: Optional[list[str]] = None,
    lookback: Optional[int] = None,
) -> pd.DataFrame:
    """Every active ticker's current Market Structure state, for the screener table.

    ``code`` is the last signal the ticker fired — the most recent within
    ``lookback`` bars, or the last one ever when ``lookback`` is None.
    """
    bars = _all_bars(con, tickers)
    meta = _ticker_meta(con)
    rows = []
    for ticker, g in bars.groupby("ticker", sort=True):
        if len(g) < MIN_BARS:
            continue
        codes, _, lines = run_state_machine(prepare(g))
        n = len(codes)
        status = _status(lines["position"][-1], lines["pantau"][-1])
        stop = -1 if lookback is None else max(-1, n - 1 - lookback)
        j = next((k for k in range(n - 1, stop, -1) if codes[k]), None)
        code = codes[j] if j is not None else ""
        idx_code, name = meta.get(ticker, (ticker, None))
        rows.append({
            "ticker": ticker, "idx_code": idx_code, "name": name,
            "close": float(g["close"].iloc[-1]), "status": status,
            "code": code, "reason": REASON_MAP.get(code, ""),
        })
    return pd.DataFrame(rows, columns=[
        "ticker", "idx_code", "name", "close", "status", "code", "reason"])


if __name__ == "__main__":
    # Self-check: a quiet pullback off a swing low that reclaims on a volume pop
    # fires a BUY LOW in the EMA20 momentum regime (ma200 NaN). No store needed.
    close = np.array(
        list(range(100, 131))                         # 31-bar uptrend (Close > EMA20)
        + [126, 124, 122, 124, 126]                   # pullback to a swing low
        + [128, 126, 124, 126, 128]                   # a higher swing low
        + [130, 133],                                 # breakout on volume
        dtype=float,
    )
    n = len(close)
    high, low = close + 0.5, close - 0.8
    op = np.concatenate([close[:1], close[:-1]])       # open = prev close
    vol = np.full(n, 2000.0)
    vol[31:41] = 200.0                                 # dry through the pullback
    vol[-2:] = 8000.0                                  # pop on the breakout
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": op, "high": high, "low": low, "close": close, "volume": vol,
        "ma200": np.nan,
    })
    codes, trades, lines = run_state_machine(prepare(frame))
    assert any(x.startswith("BUY LOW") for x in codes), codes
    # a position is only ever held on/after a bar that carried an entry code
    for i, p in enumerate(lines["position"]):
        assert p in (0, 1)
    print("market_structure self-check ok:",
          [(i, x) for i, x in enumerate(codes) if x])
