"""Point-in-time replay (IVI-77).

Walks the stored signal history as if the screener had been run live each day,
and labels what happened next.

The lookahead rules, enforced rather than assumed
------------------------------------------------
* **Indicators at date t use bars <= t.** Already true by construction — they
  are computed once, forward, and stored — and pinned by a test in
  ``test_indicators.py``.
* **The universe at date t is whoever was listed at t**, including names that
  later delisted. That is the whole reason P0 refuses to delete a ticker.
  Selecting on ``is_active`` today would quietly drop the failures.
* **Entry fills at the next bar's open**, never the signal bar's close.
* **No full-series extrema.** Nothing here takes a max or min over a whole
  price series; every window is trailing.

The guard test
--------------
:func:`replay` is deterministic given the store, so the permanent guard in
``test_replay.py`` runs it, blanks every bar after a cutoff, runs it again, and
requires the trades that had already *finished* by the cutoff to be identical.

Trades still open at the cutoff are expected to differ — their future genuinely
changed — and demanding they match would be testing the wrong thing.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import duckdb
import pandas as pd

from .costs import CostModel
from .label import OutcomeRule, Trade, label_frame, label_one


@dataclass
class ReplaySummary:
    rule_version: str
    signals_considered: int = 0
    trades_labelled: int = 0
    skipped_no_next_bar: int = 0
    tickers: int = 0
    first_entry: Optional[dt.date] = None
    last_entry: Optional[dt.date] = None
    seconds: float = 0.0
    costs_applied: bool = False
    outcomes: dict = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            "replay",
            f"  rule            : {self.rule_version}",
            f"  costs applied   : {self.costs_applied}",
            f"  tickers         : {self.tickers}",
            f"  signals seen    : {self.signals_considered}",
            f"  trades labelled : {self.trades_labelled}",
            f"  skipped (no next bar): {self.skipped_no_next_bar}",
            f"  entry range     : {self.first_entry} -> {self.last_entry}",
            f"  runtime         : {self.seconds:.1f}s",
        ]
        if self.outcomes:
            lines.append("  outcomes:")
            for outcome, n in sorted(self.outcomes.items()):
                lines.append(f"    {outcome:<8}: {n}")
        return "\n".join(lines)


def eligible_signals(
    con: duckdb.DuckDBPyConnection,
    *,
    min_depth: int = 3,
    require_gate: bool = True,
    require_above_ma5: bool = True,
    exclude_unclean: bool = True,
    tickers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Every historical signal that would have been shown, on the day it was shown.

    Note what is *not* here: no filter on `tickers.is_active`. A company that
    delisted in 2023 was tradeable in 2022 and its signals belong in the sample.
    Filtering on today's active list is survivorship bias with extra steps.
    """
    clauses = ["s.alignment_depth >= ?"]
    params: list = [min_depth]

    if require_gate:
        clauses.append("s.rsi_gate = TRUE")
    if require_above_ma5:
        clauses.append("s.above_ma5 = TRUE")
    if exclude_unclean:
        # insufficient_history stays in: a young listing was genuinely
        # tradeable, it just could not be judged at depth 5.
        clauses.append("s.data_quality NOT IN ('stale', 'suspect')")
    if tickers:
        placeholders = ", ".join("?" for _ in tickers)
        clauses.append(f"s.ticker IN ({placeholders})")
        params.extend(tickers)

    where = " AND ".join(clauses)
    return con.execute(
        f"""
        SELECT s.ticker, s.date, s.alignment_depth, s.rsi_gate, s.tier,
               s.data_quality, s.days_since_ma5_cross
          FROM signals s
         WHERE {where}
         ORDER BY s.ticker, s.date
        """,
        params,
    ).df()


def replay(
    con: duckdb.DuckDBPyConnection,
    rule: OutcomeRule,
    *,
    costs: Optional[CostModel] = None,
    min_depth: int = 3,
    require_gate: bool = True,
    require_above_ma5: bool = True,
    cutoff: Optional[dt.date] = None,
    tickers: Optional[Sequence[str]] = None,
    computed_at: Optional[dt.datetime] = None,
) -> tuple[pd.DataFrame, ReplaySummary]:
    """Replay every eligible signal and label its outcome.

    `cutoff` blanks all bars strictly after that date, which is what the
    lookahead guard uses to simulate "we are standing on date t".
    """
    started = time.perf_counter()
    summary = ReplaySummary(rule_version=rule.version, costs_applied=costs is not None)

    signals = eligible_signals(
        con,
        min_depth=min_depth,
        require_gate=require_gate,
        require_above_ma5=require_above_ma5,
        tickers=tickers,
    )
    if cutoff is not None:
        signals = signals[signals["date"] <= cutoff]

    summary.signals_considered = len(signals)
    if signals.empty:
        summary.seconds = time.perf_counter() - started
        return pd.DataFrame(), summary

    summary.first_entry = signals["date"].min()
    summary.last_entry = signals["date"].max()

    trades: list[Trade] = []
    grouped = signals.groupby("ticker", sort=True)
    summary.tickers = grouped.ngroups

    for ticker, group in grouped:
        bars = con.execute(
            "SELECT date, open, high, low, close FROM prices "
            "WHERE ticker = ? ORDER BY date",
            [ticker],
        ).df()
        if cutoff is not None:
            bars = bars[bars["date"] <= cutoff]
        if bars.empty:
            continue

        position = {d: i for i, d in enumerate(bars["date"])}
        for entry_date in group["date"]:
            index = position.get(entry_date)
            if index is None:
                continue
            trade = label_one(bars, index, rule, ticker=ticker, costs=costs)
            if trade is None:
                summary.skipped_no_next_bar += 1
                continue
            trades.append(trade)

    frame = label_frame(trades, rule, computed_at=computed_at)
    summary.trades_labelled = len(frame)
    if not frame.empty:
        summary.outcomes = frame["outcome"].value_counts().to_dict()
    summary.seconds = time.perf_counter() - started
    return frame, summary


def replay_multi(
    con: duckdb.DuckDBPyConnection,
    rules: Sequence[OutcomeRule],
    *,
    costs: Optional[CostModel] = None,
    min_depth: int = 3,
    require_gate: bool = True,
    require_above_ma5: bool = True,
    cutoff: Optional[dt.date] = None,
    tickers: Optional[Sequence[str]] = None,
    computed_at: Optional[dt.datetime] = None,
    progress=None,
) -> dict[str, tuple[pd.DataFrame, ReplaySummary]]:
    """Label the same signal set under several outcome rules in one pass.

    :func:`replay` spends most of its time fetching each ticker's bars, not
    labelling them, so running it once per rule pays that cost again for every
    rule. Walk-forward validation needs a grid of rules over the same signals,
    which made the repetition the difference between a minute and an hour.

    The result is keyed by ``rule.version`` and is required to be identical to
    calling :func:`replay` once per rule — a test asserts exactly that, because
    a faster path that quietly labels something different is worse than a slow
    one.
    """
    if not rules:
        return {}

    started = time.perf_counter()
    summaries = {
        rule.version: ReplaySummary(
            rule_version=rule.version, costs_applied=costs is not None
        )
        for rule in rules
    }

    signals = eligible_signals(
        con,
        min_depth=min_depth,
        require_gate=require_gate,
        require_above_ma5=require_above_ma5,
        tickers=tickers,
    )
    if cutoff is not None:
        signals = signals[signals["date"] <= cutoff]

    for summary in summaries.values():
        summary.signals_considered = len(signals)

    if signals.empty:
        for summary in summaries.values():
            summary.seconds = time.perf_counter() - started
        return {
            rule.version: (pd.DataFrame(), summaries[rule.version]) for rule in rules
        }

    first, last = signals["date"].min(), signals["date"].max()
    grouped = signals.groupby("ticker", sort=True)
    for summary in summaries.values():
        summary.first_entry, summary.last_entry = first, last
        summary.tickers = grouped.ngroups

    trades: dict[str, list[Trade]] = {rule.version: [] for rule in rules}

    for done, (ticker, group) in enumerate(grouped, start=1):
        bars = con.execute(
            "SELECT date, open, high, low, close FROM prices "
            "WHERE ticker = ? ORDER BY date",
            [ticker],
        ).df()
        if cutoff is not None:
            bars = bars[bars["date"] <= cutoff]
        if bars.empty:
            continue

        position = {d: i for i, d in enumerate(bars["date"])}
        for entry_date in group["date"]:
            index = position.get(entry_date)
            if index is None:
                continue
            for rule in rules:
                trade = label_one(bars, index, rule, ticker=ticker, costs=costs)
                if trade is None:
                    summaries[rule.version].skipped_no_next_bar += 1
                    continue
                trades[rule.version].append(trade)

        if progress is not None:
            progress(done, grouped.ngroups, ticker)

    out: dict[str, tuple[pd.DataFrame, ReplaySummary]] = {}
    elapsed = time.perf_counter() - started
    for rule in rules:
        frame = label_frame(trades[rule.version], rule, computed_at=computed_at)
        summary = summaries[rule.version]
        summary.trades_labelled = len(frame)
        if not frame.empty:
            summary.outcomes = frame["outcome"].value_counts().to_dict()
        summary.seconds = elapsed
        out[rule.version] = (frame, summary)
    return out
