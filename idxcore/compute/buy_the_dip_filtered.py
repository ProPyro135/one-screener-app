"""Uptrend Buy The Dip v2 with its IHSG and ADX filters (EOD).

``buy_the_dip`` (PINESCRIPT G) with two of the Pine v2's optional filters on,
the pair that measured best over ten years:

    A. IHSG        SMA20 > SMA50 and close > SMA200 on IHSG (both)
    B. ADX         ADX(14) > 20 with DI+ > DI-

Both gate the BUYs only; exits are G's. IHSG comes from the store's `indices`
table (^JKSE), as of each stock's bar date. Filter C (a tighter first stop)
is left off: it lowered net P&L in every combination tested.

A *monitoring* screener with no validated edge.
"""

from __future__ import annotations

from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from idxcore.compute import buy_the_dip as g
from idxcore.compute.buy_the_dip import MIN_BARS, category_of, run_state_machine  # noqa: F401
from idxcore.compute.market_structure import _ticker_meta  # noqa: F401
from idxcore.compute.reversal_sniper import _rma

ADX_LEN, ADX_MIN = 14, 20.0

# IHSG's own averages, then as-of joined onto each stock bar: a stock bar on a
# day IHSG has no row reads the last IHSG bar before it.
_BARS_SQL = """
    WITH ix AS (
        SELECT date,
               row_number() OVER w AS k,
               close,
               avg(close) OVER (w ROWS 19 PRECEDING)  AS f,
               avg(close) OVER (w ROWS 49 PRECEDING)  AS s,
               avg(close) OVER (w ROWS 199 PRECEDING) AS m200
          FROM indices WHERE symbol = '^JKSE'
        WINDOW w AS (ORDER BY date)
    )
    SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume,
           coalesce(ix.k >= 200 AND ix.f > ix.s AND ix.close > ix.m200, false) AS ihsg_ok
      FROM prices p
      JOIN tickers t ON t.ticker = p.ticker
      ASOF LEFT JOIN ix ON p.date >= ix.date
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


def _adx_ok(df: pd.DataFrame) -> pd.Series:
    """Pine ``ta.dmi(14, 14)``: ADX > 20 and DI+ > DI-."""
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    up, dn = high.diff(), -low.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = _rma(tr.fillna(high - low), ADX_LEN)
    di_p = 100 * _rma(pdm, ADX_LEN) / atr
    di_m = 100 * _rma(mdm, ADX_LEN) / atr
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    adx = _rma(dx.fillna(0), ADX_LEN)
    return (adx > ADX_MIN) & (di_p > di_m)


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    df = g.prepare(history)
    df["entry_ok"] = df["ihsg_ok"].astype(bool) & _adx_ok(df)
    return df


if __name__ == "__main__":
    # The gate: G's frame with entry_ok off on the golden-cross bar -> that
    # BUY is skipped, the later dip still fires.
    n = 14
    f = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": [100.0] * n, "close": [101.0] * n, "atr": [1.0] * n,
        "uptrend": [True] * n, "golden": [False] * n, "death": [False] * n, "dip": [False] * n,
        "entry_ok": [True] * n,
    })
    f.loc[1, ["golden", "entry_ok"]] = [True, False]
    f.loc[7, "dip"] = True
    _, trades, _ = run_state_machine(f)
    assert [t["entry_code"] for t in trades] == ["BUY Dip"], trades
    # ADX: a steady climb is a strong up-trend, a flat line is not.
    up = pd.DataFrame({"close": np.linspace(100, 200, 80)})
    up["high"], up["low"] = up["close"] + 1, up["close"] - 1
    assert _adx_ok(up).iloc[-1]
    flat = up.assign(close=100.0, high=101.0, low=99.0)
    assert not _adx_ok(flat).iloc[-1]
    print("buy_the_dip_filtered self-check ok")
