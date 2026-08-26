"""Bottom Fishing strategy (BB + Donchian + MA state machine).

A faithful port of the Pine Script "BB Donchian MA Signals - Bottom Fishing".
It walks each ticker's bars as a small position state machine and emits explicit
signal codes on the bars where a transition happens:

    BUY (a1/a2/a3)      entry off a Bollinger / Donchian / new-low bottom
    BUYBACK (d1/d2)     re-entry after a take-profit (d1) or cut-loss (d2)
    TP MA5 (b1/b2)      quick profit near the upper band, exit on close < MA5
    TP MA10 (c1/c2)     trend exit after a full-band breakout, exit on close < MA10
    SELL CL (e1/e2)     cut loss on Donchian break (e1) or the -10% hard stop (e2)

Consistent with the rest of this app
-------------------------------------
The store already computes Bollinger(20,2) and Donchian(20) on **raw** close
(``indicators.bollinger`` / ``indicators.donchian``), so this reads them rather
than recomputing (invariant 1). The one thing it needs and the store does not
hold is MA10, which is a plain rolling mean computed here on the fly.

Prices are raw and unadjusted, exactly as the rest of the app treats them, so a
signal here lines up with the candles the dashboard draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from ..backtest.costs import CostConfigInvalid, CostModel, CostModelNotConfigured

CL_PCT = 0.10       # -10% hard stop
TP_RATIO = 0.90     # 90% of the upper band counts as "near"
MIN_BARS = 25
DEFAULT_COSTS_PATH = Path("config") / "costs.json"

REASON_MAP = {
    "BUY (a1)": "Rebound MA5 pasca sentuh Lower Bollinger Band (Oversold Dip)",
    "BUY (a2)": "Rebound MA5 pasca sentuh Lower Donchian (Support Test)",
    "BUY (a3)": "Rebound MA5 pasca cetak New Low Donchian (Bottom Climax)",
    "BUYBACK (d1)": "Re-entry Bullish MA5 setelah siklus Take Profit (Pullback Entry)",
    "BUYBACK (d2)": "Re-entry Bullish MA5 setelah terkena Cut Loss (Bear Trap Recovery)",
    "TP MA5 (b1)": "Take Profit: Close < MA5 setelah menyentuh >= 90% Upper BB",
    "TP MA5 (b2)": "Take Profit: Close < MA5 setelah menyentuh >= 90% Upper Donchian",
    "TP MA10 (c1)": "Take Profit Trend: Close < MA10 setelah tembus Upper BB Ekstrem",
    "TP MA10 (c2)": "Take Profit Trend: Close < MA10 setelah tembus Upper Donchian Ekstrem",
    "SELL CL (e1)": "Cut Loss: Harga jebol di bawah Lower Donchian saat Entry Bar",
    "SELL CL (e2)": "Cut Loss: Penurunan harga mencapai/melebihi -10% (Hard Stop)",
}

CATEGORY = {"BUY": "BUY", "BUYBACK": "BUYBACK", "TP": "TAKE PROFIT", "SELL": "CUT LOSS"}


def category_of(code: str) -> str:
    return CATEGORY.get(code.split(" ", 1)[0], "OTHER")


def load_cost_model(path: Path = DEFAULT_COSTS_PATH) -> Optional[CostModel]:
    """The verified cost model if present and valid, else None (gross-only)."""
    if not Path(path).exists():
        return None
    try:
        return CostModel.from_file(path)
    except (CostModelNotConfigured, CostConfigInvalid):
        return None


# ---------------------------------------------------------------------------
# prepare a ticker's bars (stored indicators + the one missing MA)
# ---------------------------------------------------------------------------

def prepare(history: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the state machine needs on top of ``read.ticker_history``.

    MA10 is not stored, so it is the one indicator computed here — on raw close,
    matching how the store computed MA5/MA20.
    """
    df = history.sort_values("date").reset_index(drop=True).copy()
    df["ma10"] = pd.to_numeric(df["close"], errors="coerce").rolling(10, min_periods=10).mean()
    df["don_mid"] = (df["donchian_upper"] + df["donchian_lower"]) / 2.0
    df["don_lower_prev"] = df["donchian_lower"].shift(1)
    return df


# ---------------------------------------------------------------------------
# the state machine
# ---------------------------------------------------------------------------

def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Walk the bars; return (code per bar, round-trip trades, per-bar lines).

    A direct port of the Pine Script's persistent-state logic. Exits are
    evaluated cut-loss first (capital preservation), then trend TP, then quick TP.
    Entry and exit fill at the signal bar's close, faithful to the Pine backtest.
    """
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    ma5 = pd.to_numeric(df["ma5"], errors="coerce").to_numpy(float)
    ma10 = pd.to_numeric(df["ma10"], errors="coerce").to_numpy(float)
    ma20 = pd.to_numeric(df["ma20"], errors="coerce").to_numpy(float)
    bb_up = pd.to_numeric(df["bb_upper"], errors="coerce").to_numpy(float)
    bb_lo = pd.to_numeric(df["bb_lower"], errors="coerce").to_numpy(float)
    don_up = pd.to_numeric(df["donchian_upper"], errors="coerce").to_numpy(float)
    don_lo = pd.to_numeric(df["donchian_lower"], errors="coerce").to_numpy(float)
    don_mid = pd.to_numeric(df["don_mid"], errors="coerce").to_numpy(float)
    don_lo_prev = pd.to_numeric(df["don_lower_prev"], errors="coerce").to_numpy(float)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n
    setup_arr = [False] * n
    cl_arr = [np.nan] * n
    bdl_arr = [np.nan] * n

    pos = 0
    entry_idx = -1
    entry_price = np.nan
    entry_date = None
    entry_code = ""
    buy_don_lower = np.nan
    cl_target = np.nan
    buy_setup = False
    buy_type = ""
    reached_tp_bb90 = reached_tp_don90 = reached_ext_bb = reached_ext_don = False
    buyback_pending = False
    last_exit = ""
    bb_dip = False

    for i in range(n):
        if np.isnan(ma20[i]) or np.isnan(don_lo_prev[i]) or np.isnan(bb_lo[i]):
            continue

        if c[i] < ma5[i] and pos == 0:
            if lo[i] < don_lo_prev[i]:
                buy_setup, buy_type = True, "a3"
            elif lo[i] <= don_lo[i]:
                buy_setup, buy_type = True, "a2"
            elif lo[i] <= bb_lo[i]:
                buy_setup, buy_type = True, "a1"

        is_bullish = c[i] > o[i]
        ma5_above_both = (ma5[i] > ma20[i]) and (ma5[i] > don_mid[i])
        std_buy = (
            buy_setup and c[i] > ma5[i] and is_bullish
            and not ma5_above_both and pos == 0
        )

        if buyback_pending and pos == 0 and c[i] < ma5[i]:
            bb_dip = True
        bb_signal = (
            buyback_pending and bb_dip and pos == 0
            and c[i] > ma5[i] and is_bullish
        )

        if pos == 1:
            if h[i] >= bb_up[i]:
                reached_ext_bb = True
            if h[i] >= don_up[i]:
                reached_ext_don = True
            if h[i] >= bb_up[i] * TP_RATIO:
                reached_tp_bb90 = True
            if h[i] >= don_up[i] * TP_RATIO:
                reached_tp_don90 = True

        after_entry = pos == 1 and i > entry_idx

        e1 = after_entry and not np.isnan(buy_don_lower) and (
            c[i] < buy_don_lower or lo[i] < buy_don_lower
        )
        e2 = after_entry and not e1 and not np.isnan(cl_target) and (
            lo[i] <= cl_target or c[i] <= cl_target
        )
        c1 = after_entry and not e1 and not e2 and reached_ext_bb and c[i] < ma10[i]
        c2 = (after_entry and not e1 and not e2 and not reached_ext_bb
              and reached_ext_don and c[i] < ma10[i])
        b1 = (after_entry and not (e1 or e2 or c1 or c2)
              and reached_tp_bb90 and c[i] < ma5[i])
        b2 = (after_entry and not (e1 or e2 or c1 or c2)
              and not reached_tp_bb90 and reached_tp_don90 and c[i] < ma5[i])

        sell_cl = e1 or e2
        sell_tp = c1 or c2 or b1 or b2

        if std_buy or bb_signal:
            if std_buy:
                codes[i] = f"BUY ({buy_type or 'a1'})"
            else:
                codes[i] = "BUYBACK (d2)" if last_exit == "CL" else "BUYBACK (d1)"
            pos, entry_idx = 1, i
            entry_price, entry_date, entry_code = c[i], dates[i], codes[i]
            buy_don_lower, cl_target = don_lo[i], c[i] * (1.0 - CL_PCT)
            buy_setup, buy_type = False, ""
            reached_tp_bb90 = reached_tp_don90 = reached_ext_bb = reached_ext_don = False
            buyback_pending = bb_dip = False
        elif sell_cl or sell_tp:
            codes[i] = (
                "SELL CL (e1)" if e1 else "SELL CL (e2)" if e2 else
                "TP MA10 (c1)" if c1 else "TP MA10 (c2)" if c2 else
                "TP MA5 (b1)" if b1 else "TP MA5 (b2)"
            )
            trades.append({
                "entry_date": entry_date, "entry_price": entry_price,
                "entry_code": entry_code, "exit_date": dates[i],
                "exit_price": c[i], "exit_code": codes[i],
                "bars_held": i - entry_idx,
                "gross_return_pct": (c[i] / entry_price - 1.0) * 100.0,
                "resolved": True,
            })
            pos, entry_idx = 0, -1
            cl_target = buy_don_lower = np.nan
            reached_tp_bb90 = reached_tp_don90 = reached_ext_bb = reached_ext_don = False
            buyback_pending = True
            last_exit = "CL" if sell_cl else "TP"
            bb_dip = sell_cl

        position_arr[i] = pos
        setup_arr[i] = buy_setup
        if pos == 1:
            cl_arr[i] = cl_target
            bdl_arr[i] = buy_don_lower

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

    lines = {"position": position_arr, "setup": setup_arr,
             "cl_target": cl_arr, "buy_don_lower": bdl_arr}
    return codes, trades, lines


def annotate(history: pd.DataFrame) -> pd.DataFrame:
    """``prepare`` + run the machine, returning the frame with a ``bf_code`` column."""
    df = prepare(history)
    codes, _, _ = run_state_machine(df)
    df["bf_code"] = codes
    return df


def trades_frame(history: pd.DataFrame) -> pd.DataFrame:
    _, trades, _ = run_state_machine(prepare(history))
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# universe screen
# ---------------------------------------------------------------------------

#: One bulk pull of every bar + the stored indicators the machine reads, so a
#: universe scan is a single query + a pandas groupby rather than ~1000 round
#: trips. ponytail: whole active universe into memory (~1.7M rows); fine here.
_BARS_SQL = """
    SELECT p.ticker, p.date, p.open, p.high, p.low, p.close,
           i.ma5, i.ma20, i.bb_upper, i.bb_lower,
           i.donchian_upper, i.donchian_lower
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


def screen(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: Optional[list[str]] = None,
    lookback: int = 5,
) -> pd.DataFrame:
    """Most recent signal per active ticker that fired within the last ``lookback`` bars."""
    bars = _all_bars(con, tickers)
    meta = _ticker_meta(con)
    results = []
    for ticker, g in bars.groupby("ticker", sort=True):
        if len(g) < MIN_BARS:
            continue
        codes, _, _ = run_state_machine(prepare(g))
        n = len(codes)
        j = next((k for k in range(n - 1, max(-1, n - 1 - lookback), -1) if codes[k]), None)
        if j is None:
            continue
        idx_code, name = meta.get(ticker, (ticker, None))
        results.append({
            "ticker": ticker, "idx_code": idx_code, "name": name,
            "close": float(g["close"].iloc[j]), "date": g["date"].iloc[j],
            "code": codes[j], "category": category_of(codes[j]),
        })
    return pd.DataFrame(results, columns=[
        "ticker", "idx_code", "name", "close", "date", "code", "category"])


# ---------------------------------------------------------------------------
# radar — current strategy state per ticker, for the Search-page screener
# ---------------------------------------------------------------------------

#: The radar states and their colours (as requested): naik = holding a position
#: (riding up), bottom = flat with a bottom setup armed (watch), tunggu = flat.
STATUS_COLOUR = {"naik": "#2962FF", "bottom": "#FB8C00", "tunggu": "#787B86"}


def _status(pos: int, setup: bool) -> str:
    if pos == 1:
        return "naik"
    if setup:
        return "bottom"
    return "tunggu"


def radar_screen(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: Optional[list[str]] = None,
    lookback: int = 5,
) -> pd.DataFrame:
    """Every active ticker's current Bottom Fishing state, for the screener table.

    ``status`` is the radar (naik / bottom / tunggu); ``code`` is the most recent
    signal within ``lookback`` bars (blank if none). ``close`` is the latest price.
    """
    bars = _all_bars(con, tickers)
    meta = _ticker_meta(con)
    rows = []
    for ticker, g in bars.groupby("ticker", sort=True):
        if len(g) < MIN_BARS:
            continue
        codes, _, lines = run_state_machine(prepare(g))
        n = len(codes)
        status = _status(lines["position"][-1], lines["setup"][-1])
        j = next((k for k in range(n - 1, max(-1, n - 1 - lookback), -1) if codes[k]), None)
        code = codes[j] if j is not None else ""
        idx_code, name = meta.get(ticker, (ticker, None))
        rows.append({
            "ticker": ticker, "idx_code": idx_code, "name": name,
            "close": float(g["close"].iloc[-1]), "status": status,
            "code": code, "reason": REASON_MAP.get(code, ""),
        })
    return pd.DataFrame(rows, columns=[
        "ticker", "idx_code", "name", "close", "status", "code", "reason"])


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

def _exit_bucket(code: str) -> str:
    if code.startswith("TP MA10"):
        return "tp_ma10"
    if code.startswith("TP MA5"):
        return "tp_ma5"
    if code.startswith("SELL CL"):
        return "cut_loss"
    return "other"


def summarise_trades(frame: pd.DataFrame, *, costs: Optional[CostModel] = None) -> dict:
    if frame.empty:
        return {"trades_total": 0, "trades_resolved": 0, "open_positions": 0}
    resolved = frame[frame["resolved"]]
    n = len(resolved)
    stats = {
        "trades_total": len(frame),
        "trades_resolved": n,
        "open_positions": int((~frame["resolved"]).sum()),
    }
    if n == 0:
        return stats
    g = resolved["gross_return_pct"]
    buckets = resolved["exit_code"].map(_exit_bucket)
    stats.update({
        "win_rate_pct": float((g > 0).mean() * 100.0),
        "avg_gross_return_pct": float(g.mean()),
        "median_gross_return_pct": float(g.median()),
        "avg_bars_held": float(resolved["bars_held"].mean()),
        "tp_ma5": int((buckets == "tp_ma5").sum()),
        "tp_ma10": int((buckets == "tp_ma10").sum()),
        "cut_loss": int((buckets == "cut_loss").sum()),
    })
    if costs is not None:
        net = [costs.net_return_pct(e, x)
               for e, x in zip(resolved["entry_price"], resolved["exit_price"])]
        stats["avg_net_return_pct"] = float(np.mean(net))
    else:
        stats["avg_net_return_pct"] = None
    return stats


@dataclass
class BacktestResult:
    ok: bool
    error: str = ""
    span_years: float = 0.0
    costs_applied: bool = False
    stats: dict = field(default_factory=dict)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)


def backtest(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: Optional[list[str]] = None,
    costs: Optional[CostModel] = None,
    min_years: float = 2.0,
) -> BacktestResult:
    """Backtest the strategy over stored history, enforcing the min-history guard."""
    row = con.execute("SELECT min(date), max(date) FROM prices").fetchone()
    if not row or row[0] is None:
        return BacktestResult(ok=False, error="No prices in the store yet.")
    span = (row[1] - row[0]).days / 365.25
    if span < min_years:
        return BacktestResult(
            ok=False, span_years=span,
            error=(f"Store spans only {span:.1f} years; the backtest needs at least "
                   f"{min_years:g}."),
        )

    frames = []
    for ticker, g in _all_bars(con, tickers).groupby("ticker", sort=True):
        if len(g) < MIN_BARS:
            continue
        _codes, trades, _ = run_state_machine(prepare(g))
        if trades:
            tdf = pd.DataFrame(trades)
            tdf.insert(0, "ticker", ticker)
            frames.append(tdf)

    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return BacktestResult(
        ok=True, span_years=span, costs_applied=costs is not None,
        stats=summarise_trades(trades, costs=costs), trades=trades,
    )


if __name__ == "__main__":
    # Self-check: a dip below the bands then a bullish reclaim fires a BUY, and a
    # following crash resolves it as a losing cut-loss. No store needed.
    n = 25
    hi = [100.5] * n + [95.0, 102.0, 92.0, 86.0]
    lo = [99.5] * n + [88.0, 99.0, 80.0, 78.0]
    op = [100.0] * n + [95.0, 96.0, 90.0, 85.0]
    cl = [100.0] * n + [90.0, 101.0, 82.0, 79.0]
    close = pd.Series(cl, dtype=float)
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n + 4, freq="D"),
        "open": op, "high": hi, "low": lo, "close": cl,
        "ma5": close.rolling(5, min_periods=5).mean(),
        "ma20": close.rolling(20, min_periods=20).mean(),
        "bb_upper": close.rolling(20, min_periods=20).mean() + 2 * close.rolling(20, min_periods=20).std(ddof=0),
        "bb_lower": close.rolling(20, min_periods=20).mean() - 2 * close.rolling(20, min_periods=20).std(ddof=0),
        "donchian_upper": pd.Series(hi).rolling(20, min_periods=20).max(),
        "donchian_lower": pd.Series(lo).rolling(20, min_periods=20).min(),
    })
    codes, trades, _ = run_state_machine(prepare(frame))
    assert any(x.startswith("BUY (") for x in codes), codes
    tdf = pd.DataFrame(trades)
    assert not tdf.empty and tdf.iloc[0]["gross_return_pct"] < 0, tdf
    s = summarise_trades(tdf, costs=None)
    assert s["avg_net_return_pct"] is None and s["trades_total"] >= 1
    print("bottom_fishing self-check ok")
