"""Read-only views over the store, for the dashboard.

The UI calls these and nothing else. It never computes an indicator, never
fetches from Yahoo, and never derives a number the store does not already hold.
That is invariant 1 enforced by construction rather than by discipline: there is
no code path from the dashboard to a calculation.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import duckdb
import pandas as pd


@dataclass
class Freshness:
    """What the store actually knows, and how old it is.

    Deliberately has no concept of "live". The original app showed a green
    `LIVE · Yahoo Finance` badge that could not know whether the fetch had
    succeeded — it was a decoration, not a measurement. This reports the last
    date we have data for, and what the last ingest run did.
    """

    latest_date: Optional[dt.date] = None
    calendar_days_old: Optional[int] = None
    tickers_with_data: int = 0
    tickers_at_latest: int = 0
    last_run_id: Optional[str] = None
    last_run_at: Optional[dt.datetime] = None
    last_run_statuses: dict = field(default_factory=dict)

    @property
    def state(self) -> str:
        """'fresh' | 'aging' | 'stale' | 'empty' — never 'live'."""
        if self.latest_date is None:
            return "empty"
        if self.calendar_days_old is None:
            return "stale"
        if self.calendar_days_old <= 4:
            return "fresh"
        if self.calendar_days_old <= 10:
            return "aging"
        return "stale"

    @property
    def headline(self) -> str:
        if self.latest_date is None:
            return "No price data in the store yet."
        return (
            f"Data as of {self.latest_date:%d %b %Y} "
            f"({self.calendar_days_old} calendar days ago)"
        )

    @property
    def failures(self) -> int:
        return sum(
            n
            for status, n in self.last_run_statuses.items()
            if status in {"rate_limited", "network_error", "empty_response",
                          "parse_error", "circuit_open"}
        )


def freshness(con: duckdb.DuckDBPyConnection) -> Freshness:
    row = con.execute(
        "SELECT max(date), count(DISTINCT ticker) FROM prices"
    ).fetchone()
    latest, n_tickers = row if row else (None, 0)

    if latest is None:
        return Freshness()

    at_latest = con.execute(
        "SELECT count(*) FROM prices WHERE date = ?", [latest]
    ).fetchone()[0]

    run = con.execute(
        """
        SELECT run_id, max(started_at)
          FROM ingest_log
         WHERE source = 'yahoo'
         GROUP BY run_id
         ORDER BY 2 DESC
         LIMIT 1
        """
    ).fetchone()

    statuses = {}
    run_id = run_at = None
    if run:
        run_id, run_at = run
        statuses = {
            s: n
            for s, n in con.execute(
                "SELECT status, count(*) FROM ingest_log WHERE run_id = ? "
                "AND ticker IS NOT NULL GROUP BY status",
                [run_id],
            ).fetchall()
        }

    return Freshness(
        latest_date=latest,
        calendar_days_old=(dt.date.today() - latest).days,
        tickers_with_data=n_tickers,
        tickers_at_latest=at_latest,
        last_run_id=run_id,
        last_run_at=run_at,
        last_run_statuses=statuses,
    )


def latest_signals(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The most recent signal row per ticker, joined to price and name.

    Everything here is stored, not derived at read time.
    """
    return con.execute(
        """
        WITH newest AS (
            SELECT ticker, max(date) AS date FROM signals GROUP BY ticker
        )
        SELECT s.ticker,
               t.idx_code,
               t.name,
               t.sector,
               t.board,
               t.is_active,
               s.date,
               i.close,
               i.ma5, i.ma20, i.ma50, i.ma100, i.ma200,
               i.rsi14,
               i.macd, i.macd_signal,
               s.alignment_depth,
               s.rsi_gate,
               s.above_ma5,
               s.days_since_ma5_cross,
               s.tier,
               s.data_quality,
               s.bars_available,
               q.is_clean,
               q.stale_price,
               q.suspicious_jump,
               q.zero_volume
          FROM newest n
          JOIN signals s     ON s.ticker = n.ticker AND s.date = n.date
          LEFT JOIN indicators i  ON i.ticker = s.ticker AND i.date = s.date
          LEFT JOIN tickers t     ON t.ticker = s.ticker
          LEFT JOIN price_quality q ON q.ticker = s.ticker AND q.date = s.date
         ORDER BY s.alignment_depth DESC, i.rsi14 DESC NULLS LAST
        """
    ).df()


def ticker_history(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    *,
    limit: int = 500,
) -> pd.DataFrame:
    """Price, indicators and signals for one ticker, oldest first.

    The indicator columns are taken from ``INDICATOR_COLUMNS`` rather than
    listed here. They were listed here once, and the list fell behind twice:
    Ichimoku and Donchian were computed and stored for a while but never
    selected, so the chart quietly drew nothing, and the same thing happened
    again the day the Bollinger and fractal bands were added. Nothing errors —
    the column simply is not in the frame, and the trace is skipped as though
    the indicator had no values.
    """
    from ..compute.indicators import INDICATOR_COLUMNS

    indicator_fields = ", ".join(
        f"i.{name}" for name in INDICATOR_COLUMNS
        if name not in ("ticker", "date", "close", "computed_at")
    )
    return con.execute(
        f"""
        SELECT p.date, p.open, p.high, p.low, p.close, p.volume,
               p.dividend, p.split_factor,
               {indicator_fields},
               s.alignment_depth, s.rsi_gate, s.tier,
               s.days_since_ma5_cross, s.data_quality
          FROM prices p
          LEFT JOIN indicators i ON i.ticker = p.ticker AND i.date = p.date
          LEFT JOIN signals s    ON s.ticker = p.ticker AND s.date = p.date
         WHERE p.ticker = ?
         ORDER BY p.date DESC
         LIMIT ?
        """,
        [ticker, limit],
    ).df().sort_values("date").reset_index(drop=True)


def universe_summary(con: duckdb.DuckDBPyConnection) -> dict:
    total, active = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE is_active) FROM tickers"
    ).fetchone()
    return {"total": total, "active": active, "inactive": total - active}


def _has_table(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    """Whether a table exists — used to prefer the slim store's cache tables."""
    return bool(
        con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [name],
        ).fetchone()[0]
    )


def quality_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    # The slim hosted store caps price_quality, so it ships the full-store counts
    # precomputed; the live query below is exact against the full local store.
    if _has_table(con, "quality_summary_cache"):
        return con.execute("SELECT * FROM quality_summary_cache").df()
    return con.execute(
        """
        SELECT 'stale_price'     AS flag, count(*) FILTER (WHERE stale_price)     AS bars FROM price_quality
        UNION ALL SELECT 'zero_volume',     count(*) FILTER (WHERE zero_volume)     FROM price_quality
        UNION ALL SELECT 'suspicious_jump', count(*) FILTER (WHERE suspicious_jump) FROM price_quality
        UNION ALL SELECT 'gap',             count(*) FILTER (WHERE gap)             FROM price_quality
        UNION ALL SELECT 'impossible_bar',  count(*) FILTER (WHERE impossible_bar)  FROM price_quality
        UNION ALL SELECT 'insufficient_history', count(*) FILTER (WHERE insufficient_history) FROM price_quality
        """
    ).df()


def measured_tier_outcomes(
    con: duckdb.DuckDBPyConnection, *, rule_version: Optional[str] = None
) -> pd.DataFrame:
    """Per-tier outcomes for the rule this project actually trades.

    Returns an empty frame when that rule has no labels. That distinction
    matters: the dashboard must say "not measured yet" only while it is true,
    and stop saying it the moment the decided rule has been backtested.

    Every number here traces to a stored label, which is what invariant 8 asks
    for — the rule was never "hide everything", it was "show only what was
    measured".

    **It must never fall back to another rule.** This used to read
    ``ORDER BY computed_at DESC LIMIT 1`` — "the most recent backtest" — which
    was correct only while `labels` held exactly one rule. Once walk-forward
    validation labelled its candidates, the newest row belonged to a +12%/-6%
    target/stop rule that IVI-79 had explicitly rejected, and the dashboard
    showed its tier A mean of -0.05% where the decided rule's is +4.88%. A
    wrong measurement presented as the right one is worse than no measurement,
    so an absent rule yields nothing rather than the nearest available thing.
    """
    # The slim hosted store ships this aggregate precomputed (the ten years of
    # labels do not travel). When the cache is present it is the answer; the live
    # query below runs against the full local store.
    if _has_table(con, "measured_outcomes_cache"):
        return con.execute(
            "SELECT * FROM measured_outcomes_cache ORDER BY tier"
        ).df()

    from ..backtest.label import DECIDED_RULE

    wanted = rule_version or DECIDED_RULE.version
    exists = con.execute(
        "SELECT 1 FROM labels WHERE rule_version = ? LIMIT 1", [wanted]
    ).fetchone()
    if not exists:
        return pd.DataFrame()

    return con.execute(
        """
        SELECT s.tier,
               count(*)                                                  AS trades,
               100.0 * avg(CASE WHEN l.return_pct > 0 THEN 1 ELSE 0 END) AS win_pct,
               avg(l.return_pct)                                         AS mean_pct,
               median(l.return_pct)                                      AS median_pct,
               median(l.mae)                                             AS median_mae,
               min(l.entry_date)                                         AS first_entry,
               max(l.entry_date)                                         AS last_entry,
               any_value(l.rule_version)                                 AS rule_version
          FROM labels l
          JOIN signals s ON s.ticker = l.ticker AND s.date = l.entry_date
         WHERE l.rule_version = ? AND l.return_pct IS NOT NULL AND s.tier IS NOT NULL
         GROUP BY s.tier ORDER BY s.tier
        """,
        [wanted],
    ).df()


def signal_counts_by_date(
    con: duckdb.DuckDBPyConnection, *, days: int = 120
) -> pd.DataFrame:
    """How many tickers qualified per day, for the recent window."""
    return con.execute(
        """
        SELECT date,
               count(*) FILTER (WHERE alignment_depth = 5 AND rsi_gate) AS depth5_gated,
               count(*) FILTER (WHERE alignment_depth >= 4 AND rsi_gate) AS depth4plus_gated,
               count(*) FILTER (WHERE alignment_depth >= 3 AND rsi_gate) AS depth3plus_gated,
               count(*) AS tickers
          FROM signals
         GROUP BY date
         ORDER BY date DESC
         LIMIT ?
        """,
        [days],
    ).df().sort_values("date").reset_index(drop=True)
