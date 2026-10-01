"""One trade log for either screener: every BUY through to its TP or CL.

The radar answers "where is each stock now". This answers the question that
follows it: *when* did that signal fire, is the trade still running, and what
has it done since. A radar row reading BUY LOW is not news if the entry was in
May — of the 340 open Market Structure positions on 2026-08-31, only 76 were
entered within the last week.

``bottom_fishing``, ``market_structure``, ``reversal_sniper``,
``pattern_breakout``, ``accumulation_breakout``, ``advanced_breakout``,
``advanced_breakout_v2``, ``early_entry``, ``sniper_vcp``, ``sniper_vcp_tpsl``,
``buy_the_dip`` and ``buy_the_dip_filtered`` are the same shape — a state machine
returning ``(codes, trades, lines)`` with identical trade records — so one
builder serves them all. Pass the module itself as ``mod``.

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
#: Closed by a cut-loss signal. Can still be a gain; read ``pl_pct``.
EXIT = "EXIT"

#: How many bars the liquidity measure averages over.
TURNOVER_BARS = 60
#: "Saham tidur" — below this average daily turnover (rupiah) the name barely
#: trades. 260 of 962 tickers sat under it on 2026-08-31.
SLEEPY_TURNOVER = 100_000_000.0

COLUMNS = [
    "ticker", "idx_code", "name", "status", "signal_date", "signal_price", "buy_date", "pb",
    "buy_price",
    "last_close", "fl_pct", "hi_price", "hi_date", "max_fl_pct", "exit_date", "exit_price",
    "pl_pct", "entry_code", "exit_code", "turnover", "avg_lots", "mcap",
]


def liquidity(con: duckdb.DuckDBPyConnection, bars: int = TURNOVER_BARS) -> pd.DataFrame:
    """Per ticker over the last ``bars`` bars: average daily value traded
    (close x volume, ``turnover``) and average daily volume in lots of 100
    shares (``lots``)."""
    frame = con.execute(
        """
        WITH r AS (
            SELECT ticker, close * volume AS v, volume,
                   row_number() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
        )
        SELECT ticker, avg(v) AS turnover, avg(volume) / 100 AS lots
          FROM r WHERE rn <= ? GROUP BY 1
        """,
        [bars],
    ).df()
    return frame.set_index("ticker")


def listed_shares(con: duckdb.DuckDBPyConnection) -> pd.Series:
    """Latest listed-share count per ticker (IDX Ringkasan Saham), for market
    cap. Empty on a store that has never imported one."""
    has = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'listed_shares'"
    ).fetchone()[0]
    if not has:
        return pd.Series(dtype=float)
    frame = con.execute(
        "SELECT ticker, arg_max(shares, as_of) AS shares FROM listed_shares GROUP BY 1"
    ).df()
    return frame.set_index("ticker")["shares"].astype(float)


def _trade_row(base: dict, t: dict, mod, g: pd.DataFrame) -> dict:
    entry = float(t["entry_price"])
    running = not t["resolved"]
    # Highest high the trade actually saw: from H+1 (the bar after the BUY
    # date) through the exit bar, the owner's definition. For a running trade the machine sets exit_date to the last bar; one
    # bought on the last bar has no peak yet (NaN).
    span = (g["date"] > t["entry_date"]) & (g["date"] <= t["exit_date"])
    # max(high, close): a few stored bars are flagged impossible_bar (close
    # above high) and are kept, never deleted; the peak must still cover the
    # close the trade actually saw.
    highs = pd.concat([pd.to_numeric(g.loc[span, "high"], errors="coerce"),
                       pd.to_numeric(g.loc[span, "close"], errors="coerce")], axis=1).max(axis=1)
    hi = float(highs.max())
    # The day Max % FL was reached: the first bar that printed that high.
    hi_date = g.at[highs.idxmax(), "date"] if highs.notna().any() else pd.NaT
    ret = float(t["gross_return_pct"])
    # Every strategy on the page decides at a close and buys at the next open,
    # so the signal came on the bar before the BUY date.
    before = g.loc[g["date"] < t["entry_date"]]
    signal_date = before["date"].iloc[-1] if len(before) else pd.NaT
    # The close the BUY was decided at; the trade itself is priced off the BUY.
    signal_price = float(before["close"].iloc[-1]) if len(before) else float("nan")

    row = dict(base)
    row.update(
        status=OPEN if running else
        (EXIT if mod.category_of(t["exit_code"]) == "CUT LOSS" else CLOSED),
        pb="B",
        signal_date=signal_date, signal_price=signal_price,
        buy_date=t["entry_date"], buy_price=entry,
        entry_code=t["entry_code"],
        hi_price=hi, hi_date=hi_date, max_fl_pct=(hi / entry - 1.0) * 100.0,
    )
    # The same number means different things either side of the exit, so it is
    # never in both columns at once: floating while open, realised once closed.
    if running:
        row["fl_pct"] = ret
    else:
        row.update(exit_date=t["exit_date"], exit_price=float(t["exit_price"]),
                   exit_code=t["exit_code"], pl_pct=ret)
    return row


def next_open_fills(trades: list[dict], g: pd.DataFrame) -> tuple[list[dict], bool]:
    """Re-fill a close-signalled machine's trades the way end-of-day data allows.

    The machine decides at each close; with EOD data the order can only go in
    for the next session, so every BUY and every TP/CL fills at the next bar's
    open (as a Pine ``strategy.entry`` / ``strategy.close`` does with
    ``process_orders_on_close = false``). A BUY signalled at the last close is
    not filled yet: the trade is dropped and ``True`` is returned, so the
    caller shows a WATCHLIST row. An exit signalled at the last close has not
    filled either: the trade stays OPEN at today's close.
    """
    dates = g["date"].to_numpy()
    opens = pd.to_numeric(g["open"], errors="coerce").to_numpy(float)
    closes = pd.to_numeric(g["close"], errors="coerce").to_numpy(float)
    pos = {d: i for i, d in enumerate(pd.to_datetime(dates))}
    last = len(dates) - 1
    out, pending_buy = [], False
    for t in trades:
        i = pos[pd.Timestamp(t["entry_date"])]
        if i == last:
            pending_buy = True
            continue
        t = dict(t, entry_date=dates[i + 1], entry_price=float(opens[i + 1]))
        j = pos[pd.Timestamp(t["exit_date"])] if t["resolved"] else last
        if t["resolved"] and j < last:
            t.update(exit_date=dates[j + 1], exit_price=float(opens[j + 1]))
            j += 1
        else:
            t.update(exit_date=dates[last], exit_price=float(closes[last]),
                     exit_code="OPEN", resolved=False)
            j = last
        t["bars_held"] = j - (i + 1)
        t["gross_return_pct"] = (t["exit_price"] / t["entry_price"] - 1.0) * 100.0
        out.append(t)
    return out, pending_buy


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
    liq = liquidity(con)
    turn, lots = liq["turnover"], liq["lots"]
    shares = listed_shares(con)
    rows: list[dict] = []

    for ticker, g in bars.groupby("ticker", sort=True):
        if len(g) < mod.MIN_BARS:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        _, trades, lines = mod.run_state_machine(mod.prepare(g))
        pending_buy = False
        if getattr(mod, "FILL_NEXT_OPEN", False):
            trades, pending_buy = next_open_fills(trades, g)
        idx_code, name = meta.get(ticker, (ticker, None))
        base = {
            "ticker": ticker, "idx_code": idx_code, "name": name,
            "last_close": float(g["close"].iloc[-1]),
            "turnover": float(turn.get(ticker, float("nan"))),
            "avg_lots": float(lots.get(ticker, float("nan"))),
            # Today's market cap: latest listed shares x last close.
            "mcap": float(shares.get(ticker, float("nan"))) * float(g["close"].iloc[-1]),
        }
        for t in trades:
            rows.append(_trade_row(base, t, mod, g))
        # Flat with a setup armed, or a BUY signalled at the last close and not
        # filled yet: nothing was bought, so it is a watchlist row rather than
        # a trade. The modules name the setup flag differently.
        armed = lines.get("pantau") or lines.get("setup")
        holding = any(not t["resolved"] for t in trades)
        if not holding and (pending_buy or (lines["position"][-1] == 0 and armed and armed[-1])):
            row = {**base, "status": WATCHLIST, "pb": "P"}
            # A BUY signalled at today's close: the signal is today, the BUY
            # fills tomorrow (H+1). Market Structure and Reversal Sniper also
            # arm setups that have not signalled yet; those carry no signal.
            # The other strategies only arm on a signal.
            if pending_buy or not getattr(mod, "FILL_NEXT_OPEN", False):
                row.update(signal_date=g["date"].iloc[-1], signal_price=float(g["close"].iloc[-1]))
            rows.append(row)

    return pd.DataFrame(rows, columns=COLUMNS)


def latest(log: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: its current state.

    ``build`` appends chronologically and puts the watchlist row last, so the
    last row of each group is where the ticker stands today.
    """
    if log.empty:
        return log
    return log.groupby("ticker", sort=False).tail(1).reset_index(drop=True)


def _check_next_open_fills() -> None:
    """next_open_fills on hand-built bars: fills move to the next open; a BUY
    at the last close is pending; an exit at the last close stays OPEN."""
    g = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=6, freq="D"),
                      "open": [10.0, 11, 12, 13, 14, 15], "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5]})
    d = g["date"].to_numpy()
    closed = {"entry_date": d[0], "entry_price": 10.5, "entry_code": "BUY", "exit_date": d[2],
              "exit_price": 12.5, "exit_code": "TP", "bars_held": 2, "gross_return_pct": 0.0,
              "resolved": True}
    exit_today = dict(closed, entry_date=d[3], exit_date=d[5])
    buy_today = dict(closed, entry_date=d[5], exit_date=d[5], resolved=False, exit_code="OPEN")
    out, pending = next_open_fills([closed, exit_today, buy_today], g)
    assert pending and len(out) == 2
    assert out[0]["entry_price"] == 11.0 and out[0]["exit_price"] == 13.0, out[0]
    assert pd.Timestamp(out[0]["entry_date"]) == pd.Timestamp(d[1]) and out[0]["bars_held"] == 2
    assert out[1]["entry_price"] == 14.0 and not out[1]["resolved"] and out[1]["exit_price"] == 15.5


def _self_check(db_path: str) -> None:
    """Assert the log's invariants against a real store. See __main__ below."""
    _check_next_open_fills()
    from idxcore.compute import (accumulation_breakout, advanced_breakout,
                                 advanced_breakout_v2, bottom_fishing, buy_the_dip,
                                 buy_the_dip_filtered, early_entry,
                                 market_structure, pattern_breakout, reversal_sniper,
                                 sniper_vcp, sniper_vcp_tpsl)

    con = duckdb.connect(db_path, read_only=True)
    for mod in (market_structure, reversal_sniper, pattern_breakout, accumulation_breakout,
                advanced_breakout, early_entry, buy_the_dip, buy_the_dip_filtered,
                sniper_vcp, advanced_breakout_v2, sniper_vcp_tpsl, bottom_fishing):
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
        # the price the trade exited at. Except where part of the position is
        # sold on the BUY bar itself, which the H+1 peak does not cover.
        if not getattr(mod, "PARTIAL_EXIT_ON_ENTRY_BAR", False):
            assert (traded["max_fl_pct"].isna()
                    | (traded["max_fl_pct"] >= traded["pl_pct"].fillna(-1e9) - 1e-9)).all()
        # H+1 onward: no peak for a trade bought on the last bar, or one that
        # exited on its own entry bar (Reversal Sniper checks its CL there).
        no_peak = traded[traded["max_fl_pct"].isna()]
        # The peak's date sits inside the trade: after the BUY, by the exit.
        assert traded["hi_date"].isna().equals(traded["max_fl_pct"].isna())
        peaked = traded[traded["hi_date"].notna()]
        end = peaked["exit_date"].fillna(pd.Timestamp.max)
        assert ((peaked["hi_date"] > peaked["buy_date"]) & (peaked["hi_date"] <= end)).all()
        assert (no_peak["status"].eq(OPEN) | no_peak["exit_date"].eq(no_peak["buy_date"])).all()
        # The signal is the close before the BUY's fill.
        signalled = traded[traded["signal_date"].notna()]
        assert (signalled["signal_date"] < signalled["buy_date"]).all()
        assert signalled["signal_price"].notna().all()
        # A watchlist row with a signal is today's signal, not bought yet.
        watch = log[(log["status"] == WATCHLIST) & log["signal_date"].notna()]
        assert watch["buy_date"].isna().all() and watch["signal_price"].notna().all()
        # EXIT is the cut-loss bucket by construction. Not necessarily a loss:
        # Bottom Fishing's stops trigger on the bar's low but fill at its close,
        # so a bar that dips through the stop and recovers exits in profit.
        assert (cur[cur["status"] == EXIT]["exit_code"].map(mod.category_of) == "CUT LOSS").all()
        # The owner's rule for the Pine strategies: a take-profit is a profit.
        # A strategy whose TP fills at the next open is profitable at the
        # signal but can gap under the entry by the fill, so it is exempt.
        next_open = getattr(mod, "TP_FILLS_NEXT_OPEN", False) or getattr(mod, "FILL_NEXT_OPEN", False)
        if mod is not bottom_fishing and not next_open:
            assert (log.loc[log["status"] == CLOSED, "pl_pct"] > 0).all()
        print(f"{mod.__name__}: {len(log)} trades, {len(cur)} tickers, "
              f"{dict(cur['status'].value_counts())}")
    print("trade_log self-check OK")


#: Table the nightly publish writes into the slim store (see ``publish``).
CACHE_TABLE = "trade_log_cache"


def publish(full_path: str, slim_path: str) -> int:
    """Write every trade of every Pine strategy, over the FULL history, into
    the slim store as ``trade_log_cache``.

    The hosted app only carries the last ~180 bars, too short for a backtest
    and too short for the strategies to warm up; walking ten years per request
    would also lag the page. So the nightly job computes the log here, against
    the full store, and the page just reads the finished rows.

    Every ticker is walked, delisted ones too — a backtest must not drop the
    names that failed. ``is_active`` travels with each row so the current-
    status view can still hide them.
    """
    from idxcore.compute import (accumulation_breakout, advanced_breakout,
                                 advanced_breakout_v2, buy_the_dip, buy_the_dip_filtered,
                                 early_entry, market_structure,
                                 pattern_breakout, reversal_sniper, sniper_vcp,
                                 sniper_vcp_tpsl)

    strategies = {"A": market_structure, "B": reversal_sniper, "C": pattern_breakout,
                  "D": accumulation_breakout, "E": advanced_breakout, "F": early_entry,
                  "G": buy_the_dip, "H": buy_the_dip_filtered, "I": sniper_vcp,
                  "J": advanced_breakout_v2, "K": sniper_vcp_tpsl}
    full = duckdb.connect(full_path, read_only=True)
    try:
        active = dict(full.execute("SELECT ticker, is_active FROM tickers").fetchall())
        frames = [
            build(full, mod, tickers=list(active)).assign(strategy=key)
            for key, mod in strategies.items()
        ]
    finally:
        full.close()
    log = pd.concat(frames, ignore_index=True)
    log["is_active"] = log["ticker"].map(active).fillna(False).astype(bool)
    # latest() relies on row order (chronological, watchlist last); SQL does
    # not promise it back, so it travels as a column.
    log["seq"] = range(len(log))

    slim = duckdb.connect(slim_path)
    try:
        slim.register("log_df", log)
        slim.execute(f"CREATE OR REPLACE TABLE {CACHE_TABLE} AS SELECT * FROM log_df")
    finally:
        slim.close()
    return len(log)


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) == 4 and sys.argv[1] == "--publish":
        print(f"{CACHE_TABLE}: {publish(sys.argv[2], sys.argv[3])} rows written")
    else:
        _self_check(sys.argv[1] if len(sys.argv) > 1 else "data/idx_slim.duckdb")
