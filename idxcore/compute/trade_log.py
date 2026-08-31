"""One trade log for either screener: every BUY through to its TP or CL.

The radar answers "where is each stock now". This answers the question that
follows it: *when* did that signal fire, is the trade still running, and what
has it done since. A radar row reading BUY LOW is not news if the entry was in
May — of the 340 open Market Structure positions on 2026-08-31, only 76 were
entered within the last week.

``bottom_fishing`` and ``market_structure`` are the same shape — a state machine
returning ``(codes, trades, lines)`` with identical trade records — so one
builder serves both. Pass the module itself as ``mod``.

Nothing here re-derives a signal; it reads the trades the state machine already
produced. The peak price behind *Hi Prices* is taken from the stored bars
between entry and exit rather than tracked inside the machine, so neither state
machine is touched and no backtest number can move.

Returns are **gross**. The cost model lives in ``config/costs.json``, which is
not deployed with the app, and the fee rates in it are assumed rather than taken
from a real confirmation (HANDOFF §5). Label the column accordingly; do not
present these as net.
"""

from __future__ import annotations

from typing import Optional

import duckdb
import pandas as pd

#: A position is running; floating P/L is in ``fl_pct``.
OPEN = "OPEN"
#: A setup is armed but nothing has been bought — no entry price yet.
WATCHLIST = "WATCHLIST"
#: Closed by a take-profit signal. Can still be a loss; read ``pl_pct``.
CLOSED = "CLOSED"
#: Closed by a cut-loss signal.
EXIT = "EXIT"

#: How many bars the liquidity measure averages over.
TURNOVER_BARS = 60
#: "Saham tidur" — below this average daily turnover (rupiah) the name barely
#: trades. 260 of 962 tickers sat under it on 2026-08-31.
SLEEPY_TURNOVER = 100_000_000.0

COLUMNS = [
    "ticker", "idx_code", "name", "status", "buy_date", "pb", "buy_price",
    "last_close", "fl_pct", "hi_price", "max_fl_pct", "exit_date", "exit_price",
    "pl_pct", "entry_code", "exit_code", "turnover",
]


def turnover(con: duckdb.DuckDBPyConnection, bars: int = TURNOVER_BARS) -> pd.Series:
    """Average daily close x volume per ticker over the last ``bars`` bars.

    The store has no market cap or share count, so this is the only size /
    liquidity measure available to it.
    """
    frame = con.execute(
        """
        WITH r AS (
            SELECT ticker, close * volume AS v,
                   row_number() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
        )
        SELECT ticker, avg(v) AS turnover FROM r WHERE rn <= ? GROUP BY 1
        """,
        [bars],
    ).df()
    return frame.set_index("ticker")["turnover"]


def _trade_row(base: dict, t: dict, mod, g: pd.DataFrame) -> dict:
    entry = float(t["entry_price"])
    running = not t["resolved"]
    # Highest high the trade actually saw. For a running trade the machine sets
    # exit_date to the last bar, so this spans entry -> today.
    span = (g["date"] >= t["entry_date"]) & (g["date"] <= t["exit_date"])
    hi = float(pd.to_numeric(g.loc[span, "high"], errors="coerce").max())
    ret = float(t["gross_return_pct"])

    row = dict(base)
    row.update(
        status=OPEN if running else
        (EXIT if mod.category_of(t["exit_code"]) == "CUT LOSS" else CLOSED),
        pb="B",
        buy_date=t["entry_date"], buy_price=entry, entry_code=t["entry_code"],
        hi_price=hi, max_fl_pct=(hi / entry - 1.0) * 100.0,
    )
    # The same number means different things either side of the exit, so it is
    # never in both columns at once: floating while open, realised once closed.
    if running:
        row["fl_pct"] = ret
    else:
        row.update(exit_date=t["exit_date"], exit_price=float(t["exit_price"]),
                   exit_code=t["exit_code"], pl_pct=ret)
    return row


def build(
    con: duckdb.DuckDBPyConnection,
    mod,
    *,
    tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Every trade of every active ticker, oldest first within each ticker.

    One row per trade, plus one WATCHLIST row for a ticker that is flat with a
    setup armed right now. ``latest(...)`` reduces this to one row per ticker.
    """
    bars = mod._all_bars(con, tickers)
    meta = mod._ticker_meta(con)
    turn = turnover(con)
    rows: list[dict] = []

    for ticker, g in bars.groupby("ticker", sort=True):
        if len(g) < mod.MIN_BARS:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        _, trades, lines = mod.run_state_machine(mod.prepare(g))
        idx_code, name = meta.get(ticker, (ticker, None))
        base = {
            "ticker": ticker, "idx_code": idx_code, "name": name,
            "last_close": float(g["close"].iloc[-1]),
            "turnover": float(turn.get(ticker, float("nan"))),
        }
        for t in trades:
            rows.append(_trade_row(base, t, mod, g))
        # Flat with a setup armed: nothing was bought, so it is a watchlist row
        # rather than a trade. The two modules name the flag differently.
        armed = lines.get("pantau") or lines.get("setup")
        if lines["position"][-1] == 0 and armed and armed[-1]:
            rows.append({**base, "status": WATCHLIST, "pb": "P"})

    return pd.DataFrame(rows, columns=COLUMNS)


def latest(log: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: its current state.

    ``build`` appends chronologically and puts the watchlist row last, so the
    last row of each group is where the ticker stands today.
    """
    if log.empty:
        return log
    return log.groupby("ticker", sort=False).tail(1).reset_index(drop=True)


def _self_check(db_path: str) -> None:
    """Assert the log's invariants against a real store. See __main__ below."""
    from idxcore.compute import bottom_fishing, market_structure

    con = duckdb.connect(db_path, read_only=True)
    for mod in (market_structure, bottom_fishing):
        log = build(con, mod)
        cur = latest(log)
        traded = log[log["status"] != WATCHLIST]

        assert len(cur) == log["ticker"].nunique(), "latest() must keep every ticker once"
        # The two return columns are never both filled: one is floating, the
        # other realised, and which one applies is what `status` means.
        assert cur[cur["status"] == OPEN]["pl_pct"].isna().all()
        assert cur[cur["status"].isin([CLOSED, EXIT])]["fl_pct"].isna().all()
        # A watchlist row is the "armed, nothing bought" case: no entry at all.
        assert (cur[cur["status"] == WATCHLIST]["pb"] == "P").all()
        assert (cur[cur["status"] != WATCHLIST]["pb"] == "B").all()
        assert cur[cur["status"] == WATCHLIST]["buy_price"].isna().all()
        # The peak is taken over the trade's own bars, so it cannot sit below
        # the price the trade exited at.
        assert (traded["max_fl_pct"] >= traded["pl_pct"].fillna(-1e9) - 1e-9).all()
        # EXIT is the cut-loss bucket by construction.
        assert (cur[cur["status"] == EXIT]["pl_pct"] < 0).all()
        print(f"{mod.__name__}: {len(log)} trades, {len(cur)} tickers, "
              f"{dict(cur['status'].value_counts())}")
    print("trade_log self-check OK")


if __name__ == "__main__":  # pragma: no cover
    import sys

    _self_check(sys.argv[1] if len(sys.argv) > 1 else "data/idx_slim.duckdb")
