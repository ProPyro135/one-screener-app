"""Cross-checking stored prices (IVI-85).

What the second source turned out to be
---------------------------------------
Every candidate the issue names was probed on 2026-08-21. None is reachable
from a script:

===================  ====================================================
IDX official site    HTTP 403, Cloudflare interstitial. A real browser gets
                     through; a script does not, and defeating a bot check
                     is out of scope.
Sectors.app          v1 discontinued 2026-05-11. v2 exists but returns 403
                     without an API key, which means an account signup.
NeaByteLab/IDX-API   ``api-idx.neabyte.com`` no longer resolves.
Stooq                Serves a JavaScript bot-check page on .com and .pl,
                     for ``.jk``, ``.id`` and index symbols alike.
===================  ====================================================

Also probed, outside the issue's list: Alpha Vantage and Twelve Data both work
but require a free API key, and MarketWatch is bot-blocked.

So **there is no genuinely independent free source available to a script
today**. Rather than pretend otherwise, `verify` does the two checks that are
actually possible, and is explicit about what each one can and cannot catch.

Check 1: OHLC integrity
-----------------------
Pure arithmetic on stored bars, no network. A bar where the high is below the
close, or the low above the open, is impossible — it is corruption, and no
second source is needed to know that. This catches a class of error the quality
gates miss entirely, because they look at *changes between* bars rather than at
the internal consistency of one.

Check 2: re-fetch and compare
-----------------------------
Pull a sample of tickers from Yahoo again and diff against what is stored.

**Be honest about what this is worth.** Yahoo compared against Yahoo cannot
catch an error Yahoo itself is making — the ELTY-style 10x print would be
returned identically both times. What it does catch: ingestion and parsing
bugs on our side, silent schema drift in yfinance, and Yahoo revising a bar
after we stored it. That is a narrower claim than "second source", and it is
the claim that is actually true.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, Sequence

import duckdb
import numpy as np
import pandas as pd

from . import yahoo


@dataclass
class VerifyReport:
    ohlc_violations: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_mismatches: pd.DataFrame = field(default_factory=pd.DataFrame)
    tickers_compared: int = 0
    bars_compared: int = 0
    refetch_failures: dict = field(default_factory=dict)
    tolerance_pct: float = 0.5

    @property
    def clean(self) -> bool:
        return self.ohlc_violations.empty and self.price_mismatches.empty

    def render(self) -> str:
        lines = [
            "verify",
            f"  tolerance        : {self.tolerance_pct}%",
            "",
            "  OHLC integrity (no network, catches impossible bars)",
            f"    violations     : {len(self.ohlc_violations)}",
        ]
        if not self.ohlc_violations.empty:
            for _, r in self.ohlc_violations.head(5).iterrows():
                lines.append(f"      {r['ticker']} {r['date']} — {r['problem']}")
            if len(self.ohlc_violations) > 5:
                lines.append(f"      ... and {len(self.ohlc_violations) - 5} more")

        lines += [
            "",
            "  Re-fetch comparison (catches OUR bugs, not Yahoo's)",
            f"    tickers        : {self.tickers_compared}",
            f"    bars compared  : {self.bars_compared}",
            f"    mismatches     : {len(self.price_mismatches)}",
        ]
        if not self.price_mismatches.empty:
            for _, r in self.price_mismatches.head(5).iterrows():
                lines.append(
                    f"      {r['ticker']} {r['date']} — stored {r['stored']:g} "
                    f"vs fetched {r['fetched']:g} ({r['diff_pct']:+.2f}%)"
                )
            if len(self.price_mismatches) > 5:
                lines.append(f"      ... and {len(self.price_mismatches) - 5} more")
        if self.refetch_failures:
            lines.append(f"    fetch failures : {self.refetch_failures}")
        return "\n".join(lines)


def check_ohlc_integrity(
    con: duckdb.DuckDBPyConnection, *, tickers: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Bars that cannot exist, whatever the source says.

    high must be the highest of the four, low the lowest, and neither can be
    non-positive. No second source is needed to know a bar breaking that is
    wrong.
    """
    where = ""
    params: list = []
    if tickers:
        where = f"WHERE ticker IN ({', '.join('?' for _ in tickers)})"
        params = list(tickers)

    # A subquery rather than QUALIFY: DuckDB's QUALIFY requires a window
    # function to be present, and there is none here.
    return con.execute(
        f"""
        SELECT * FROM (
            SELECT ticker, date, open, high, low, close,
                   CASE
                     WHEN high < low                  THEN 'high below low'
                     WHEN high < open OR high < close THEN 'high below open/close'
                     WHEN low > open OR low > close   THEN 'low above open/close'
                     WHEN low <= 0 OR close <= 0      THEN 'non-positive price'
                     ELSE 'ok'
                   END AS problem
              FROM prices
              {where}
        )
        WHERE problem <> 'ok'
        ORDER BY ticker, date
        """,
        params,
    ).df()


def compare_against_refetch(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: Sequence[str],
    days: int = 30,
    tolerance_pct: float = 0.5,
    downloader=None,
) -> tuple[pd.DataFrame, int, dict]:
    """Re-fetch recent bars and diff them against what is stored."""
    start = dt.date.today() - dt.timedelta(days=days)
    result = yahoo.fetch_many(
        list(tickers),
        start=start,
        cfg=yahoo.BatchConfig(batch_size=20, max_workers=1, pause_between_batches=0.5),
        downloader=downloader,
    )

    failures = {
        s: n for s, n in result.status_counts().items() if s != "ok"
    }
    if result.frame.empty:
        return pd.DataFrame(), 0, failures

    fetched = result.frame.loc[:, ["ticker", "date", "close"]].rename(
        columns={"close": "fetched"}
    )
    con.register("_refetch", fetched)
    try:
        merged = con.execute(
            """
            SELECT f.ticker, f.date, p.close AS stored, f.fetched
              FROM _refetch f
              JOIN prices p ON p.ticker = f.ticker AND p.date = f.date
             WHERE p.close IS NOT NULL AND f.fetched IS NOT NULL
            """
        ).df()
    finally:
        con.unregister("_refetch")

    if merged.empty:
        return pd.DataFrame(), 0, failures

    merged["diff_pct"] = (merged["fetched"] / merged["stored"] - 1.0) * 100.0
    mismatches = merged[merged["diff_pct"].abs() > tolerance_pct].copy()
    return mismatches.sort_values("diff_pct", key=np.abs, ascending=False), len(merged), failures


def run(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: Optional[Sequence[str]] = None,
    sample: int = 25,
    days: int = 30,
    tolerance_pct: float = 0.5,
    skip_refetch: bool = False,
    downloader=None,
) -> VerifyReport:
    report = VerifyReport(tolerance_pct=tolerance_pct)
    report.ohlc_violations = check_ohlc_integrity(con, tickers=tickers)

    if skip_refetch:
        return report

    if tickers:
        chosen = list(tickers)
    else:
        # Sample rather than sweep: this is meant to run weekly, and hammering
        # Yahoo with the whole universe to check it is the behaviour the
        # ingestion layer exists to avoid.
        chosen = [
            r[0]
            for r in con.execute(
                "SELECT ticker FROM tickers WHERE is_active "
                "ORDER BY random() LIMIT ?",
                [sample],
            ).fetchall()
        ]

    report.tickers_compared = len(chosen)
    (
        report.price_mismatches,
        report.bars_compared,
        report.refetch_failures,
    ) = compare_against_refetch(
        con,
        tickers=chosen,
        days=days,
        tolerance_pct=tolerance_pct,
        downloader=downloader,
    )
    return report
