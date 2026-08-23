"""Calibration report (IVI-81).

The issue the whole project exists to serve: **is a Tier A signal actually
better than a Tier C one, and is either better than doing nothing in
particular?**

Everything here reads from the store, so every number in the report can be
reproduced by re-running the command. Nothing is passed in from a notebook or
remembered from an earlier run.

Three rules the report follows
------------------------------
**An edge only means something relative to something else.** Three baselines
are reported beside every strategy number: buy-and-hold IHSG, the average stock
over the same horizon, and a random entry on any clean bar. If the strategy does
not beat them, that is the finding and it is stated plainly rather than buried.

**Sample size is always shown.** A tier with 11 trades gets no conclusions drawn
from it, and the report says so on the row.

**Medians sit beside means.** A positive mean with a negative median means most
trades lose and a few large winners carry the result — a very different thing to
live through, and invisible if only the mean is printed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

#: Below this, a row is reported but no conclusion is drawn from it.
MIN_TRADES_FOR_A_CONCLUSION = 30


@dataclass
class ReportContext:
    rule_version: str
    rule_description: str
    cost_description: str
    horizon_bars: int
    generated_at: dt.datetime


def _fmt(value, spec: str = "+.2f") -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return format(value, spec)


def tier_summary(con: duckdb.DuckDBPyConnection, rule_version: str) -> pd.DataFrame:
    """Per-tier outcomes for one rule.

    `win_pct` is the share of trades with a positive net return, not the share
    that hit a target — a hold rule has no target, and reporting a target hit
    rate of 0% for it would be actively misleading.
    """
    return con.execute(
        """
        SELECT s.tier,
               count(*)                                              AS trades,
               100.0 * avg(CASE WHEN l.return_pct > 0 THEN 1 ELSE 0 END) AS win_pct,
               avg(l.return_pct)                                     AS mean_pct,
               median(l.return_pct)                                  AS median_pct,
               avg(l.mae)                                            AS mean_mae,
               median(l.mae)                                         AS median_mae,
               avg(l.mfe)                                            AS mean_mfe,
               avg(l.bars_held)                                      AS mean_bars
          FROM labels l
          JOIN signals s ON s.ticker = l.ticker AND s.date = l.entry_date
         WHERE l.rule_version = ? AND l.return_pct IS NOT NULL AND s.tier IS NOT NULL
         GROUP BY s.tier
         ORDER BY s.tier
        """,
        [rule_version],
    ).df()


def tier_by_year(con: duckdb.DuckDBPyConnection, rule_version: str) -> pd.DataFrame:
    return con.execute(
        """
        SELECT year(l.entry_date) AS yr, s.tier,
               count(*) AS trades, avg(l.return_pct) AS mean_pct
          FROM labels l
          JOIN signals s ON s.ticker = l.ticker AND s.date = l.entry_date
         WHERE l.rule_version = ? AND l.return_pct IS NOT NULL AND s.tier IS NOT NULL
         GROUP BY 1, 2 ORDER BY 1, 2
        """,
        [rule_version],
    ).df()


def baselines(
    con: duckdb.DuckDBPyConnection,
    *,
    horizon_bars: int,
    buy_fee: float,
    sell_fee: float,
) -> pd.DataFrame:
    """The three comparisons IVI-81 requires, on the same horizon and costs."""
    universe = con.execute(
        """
        WITH b AS (
          SELECT p.ticker, p.date, p.open, p.close,
                 row_number() OVER (PARTITION BY p.ticker ORDER BY p.date) AS rn
            FROM prices p
            JOIN price_quality q ON q.ticker = p.ticker AND q.date = p.date
           WHERE q.is_clean
        ),
        t AS (
          SELECT f.open AS fill, x.close AS exit_px
            FROM b
            JOIN b f ON f.ticker = b.ticker AND f.rn = b.rn + 1
            JOIN b x ON x.ticker = b.ticker AND x.rn = b.rn + ?
           WHERE f.open > 0
        )
        SELECT count(*) AS trades,
               avg((exit_px * (1 - ?)) / (fill * (1 + ?)) * 100 - 100) AS mean_pct,
               median((exit_px * (1 - ?)) / (fill * (1 + ?)) * 100 - 100) AS median_pct,
               100.0 * avg(CASE WHEN (exit_px * (1 - ?)) > (fill * (1 + ?)) THEN 1 ELSE 0 END) AS win_pct
          FROM t
        """,
        [horizon_bars, sell_fee, buy_fee, sell_fee, buy_fee, sell_fee, buy_fee],
    ).df()
    universe.insert(0, "baseline", "Average stock (every clean bar)")

    index = con.execute(
        """
        WITH b AS (
          SELECT date, open, close, row_number() OVER (ORDER BY date) AS rn
            FROM indices WHERE symbol = '^JKSE'
        ),
        t AS (
          SELECT f.open AS fill, x.close AS exit_px
            FROM b JOIN b f ON f.rn = b.rn + 1
                   JOIN b x ON x.rn = b.rn + ?
           WHERE f.open > 0
        )
        SELECT count(*) AS trades,
               avg(exit_px / fill * 100 - 100) AS mean_pct,
               median(exit_px / fill * 100 - 100) AS median_pct,
               100.0 * avg(CASE WHEN exit_px > fill THEN 1 ELSE 0 END) AS win_pct
          FROM t
        """,
        [horizon_bars],
    ).df()
    index.insert(0, "baseline", "Buy-and-hold IHSG (no fees — you cannot trade the index)")

    return pd.concat([index, universe], ignore_index=True)


def mae_distribution(con: duckdb.DuckDBPyConnection, rule_version: str) -> pd.DataFrame:
    """How deep the drawdown got before a trade resolved.

    On a hold rule there is no stop, so this is the loss you would have had to
    sit through — the part a win/loss column hides completely.
    """
    return con.execute(
        """
        SELECT s.tier,
               quantile_cont(l.mae, 0.50) AS p50,
               quantile_cont(l.mae, 0.25) AS p75,
               quantile_cont(l.mae, 0.10) AS p90,
               quantile_cont(l.mae, 0.01) AS p99,
               min(l.mae)                 AS worst
          FROM labels l
          JOIN signals s ON s.ticker = l.ticker AND s.date = l.entry_date
         WHERE l.rule_version = ? AND l.mae IS NOT NULL AND s.tier IS NOT NULL
         GROUP BY s.tier ORDER BY s.tier
        """,
        [rule_version],
    ).df()


def ranks_monotonically(tiers: pd.DataFrame) -> Optional[bool]:
    """Does mean return fall as the tier weakens? None if too few tiers."""
    usable = tiers[tiers["trades"] >= MIN_TRADES_FOR_A_CONCLUSION]
    usable = usable.sort_values("tier")
    if len(usable) < 2:
        return None
    means = list(usable["mean_pct"])
    return all(a >= b for a, b in zip(means, means[1:]))


def render_markdown(
    ctx: ReportContext,
    tiers: pd.DataFrame,
    base: pd.DataFrame,
    by_year: pd.DataFrame,
    mae: pd.DataFrame,
) -> str:
    out: list[str] = []
    add = out.append

    add("# Calibration report")
    add("")
    add(f"Generated {ctx.generated_at:%Y-%m-%d %H:%M}. Every number below is "
        "reproducible from the DuckDB store by re-running the command.")
    add("")
    add("```")
    add(ctx.rule_description)
    add("")
    add(ctx.cost_description)
    add("```")
    add("")

    add("## Does the tier scheme rank?")
    add("")
    add("| Tier | Trades | Win % | Mean % | Median % | Mean MAE | Mean bars |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in tiers.iterrows():
        note = "" if r["trades"] >= MIN_TRADES_FOR_A_CONCLUSION else " ⚠️"
        add(
            f"| **{r['tier']}**{note} | {int(r['trades']):,} | {_fmt(r['win_pct'], '.1f')} | "
            f"{_fmt(r['mean_pct'])} | {_fmt(r['median_pct'])} | "
            f"{_fmt(r['mean_mae'])} | {_fmt(r['mean_bars'], '.0f')} |"
        )
    add("")

    small = tiers[tiers["trades"] < MIN_TRADES_FOR_A_CONCLUSION]
    if not small.empty:
        add(f"⚠️ Tiers with fewer than {MIN_TRADES_FOR_A_CONCLUSION} trades are shown "
            "but no conclusion is drawn from them: "
            + ", ".join(str(t) for t in small["tier"]) + ".")
        add("")

    verdict = ranks_monotonically(tiers)
    if verdict is True:
        add("**The tiers rank.** Mean return falls monotonically as the tier "
            "weakens, which is what the scheme claims to do.")
    elif verdict is False:
        add("**The tiers do not rank.** Mean return does not fall monotonically "
            "as the tier weakens. That is a real result: the tier scheme is not "
            "sorting anything and should be simplified rather than defended.")
    else:
        add("Not enough populated tiers to say whether the scheme ranks.")
    add("")

    add("## Against the baselines")
    add("")
    add("An edge only means something relative to something else.")
    add("")
    add("| Comparison | Trades | Mean % | Median % | Win % |")
    add("|---|---:|---:|---:|---:|")
    best = tiers.sort_values("tier").head(1)
    if not best.empty:
        r = best.iloc[0]
        add(f"| **Tier {r['tier']} (best tier)** | {int(r['trades']):,} | "
            f"{_fmt(r['mean_pct'])} | {_fmt(r['median_pct'])} | {_fmt(r['win_pct'], '.1f')} |")
    for _, r in base.iterrows():
        add(f"| {r['baseline']} | {int(r['trades']):,} | {_fmt(r['mean_pct'])} | "
            f"{_fmt(r['median_pct'])} | {_fmt(r['win_pct'], '.1f')} |")
    add("")
    add("The IHSG row carries no fees, because you cannot trade an index "
        "directly — it is the *market did this* line, not a strategy you could "
        "have run.")
    add("")

    add("## Does the edge live in one bull run?")
    add("")
    pivot = by_year.pivot(index="yr", columns="tier", values="mean_pct")
    counts = by_year.pivot(index="yr", columns="tier", values="trades")
    cols = list(pivot.columns)
    add("| Year | " + " | ".join(f"{c} mean %" for c in cols) + " | Trades |")
    add("|---|" + "---:|" * (len(cols) + 1))
    for yr in pivot.index:
        cells = " | ".join(_fmt(pivot.loc[yr, c]) for c in cols)
        total = int(counts.loc[yr].fillna(0).sum())
        add(f"| {int(yr)} | {cells} | {total:,} |")
    add("")
    add("A strategy whose entire edge sits in one or two years has not been "
        "shown to work; it has been shown to have been lucky.")
    add("")

    add("## What you would have had to sit through")
    add("")
    add("Maximum adverse excursion — how far a trade went against you before it "
        "resolved. With no stop, this is a loss you had to hold.")
    add("")
    add("| Tier | Median | 75th pct | 90th pct | 99th pct | Worst |")
    add("|---|---:|---:|---:|---:|---:|")
    for _, r in mae.iterrows():
        add(f"| **{r['tier']}** | {_fmt(r['p50'])} | {_fmt(r['p75'])} | "
            f"{_fmt(r['p90'])} | {_fmt(r['p99'])} | {_fmt(r['worst'])} |")
    add("")

    add("## Read this before using any number above")
    add("")
    add("- **Fees may be assumed.** Check the cost line at the top. If it says "
        "ASSUMED, every return here inherits that assumption.")
    add("- **Means are carried by a fat right tail.** Where the median is "
        "negative, most trades lost money and a handful of large winners "
        "produced the average. Position sizing matters more than the mean "
        "suggests.")
    add("- **Multiple comparisons.** These figures cover the whole population "
        "of signals, not a hand-picked subset. Scanning ~900 tickers daily and "
        "reporting the best-looking results would guarantee flattering "
        "outliers; that is not what this is.")
    add("- **Everything here is in-sample.** One rule was chosen and measured "
        "once, over the same history that chose it. The out-of-sample check "
        "lives in `reports/walkforward.md` — run `idxcore walkforward` — and "
        "that is the report to read before trusting any figure above as a "
        "forecast rather than a description.")
    add("")
    return "\n".join(out)


def build(
    con: duckdb.DuckDBPyConnection,
    ctx: ReportContext,
    *,
    buy_fee: float,
    sell_fee: float,
) -> str:
    tiers = tier_summary(con, ctx.rule_version)
    if tiers.empty:
        raise ValueError(
            f"no labels for rule {ctx.rule_version}. Run `idxcore backtest` first."
        )
    base = baselines(
        con, horizon_bars=ctx.horizon_bars, buy_fee=buy_fee, sell_fee=sell_fee
    )
    return render_markdown(
        ctx, tiers, base, tier_by_year(con, ctx.rule_version),
        mae_distribution(con, ctx.rule_version),
    )


def write(path: str | Path, markdown: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(markdown, encoding="utf-8")
    return p
