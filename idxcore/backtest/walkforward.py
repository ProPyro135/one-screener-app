"""Walk-forward validation.

The calibration report ends with a warning it could not act on: *"One rule was
chosen and measured once. Nothing here is out-of-sample."* This module is what
acts on it.

What is actually being tested
-----------------------------
Not "does the strategy work" — the calibration report already measures that on
the full history. The question here is narrower and more uncomfortable: **how
much of that measured edge is an artefact of having chosen the rule after
seeing the answer?**

The exit rule was picked on evidence from the whole ten years (IVI-79): a -4%
stop sat inside the median adverse excursion, so it fired on noise, and a target
capped the fat right tail. That reasoning is sound and the decision stands. But
it is still a decision taken with the full sample in view, and a decision like
that cannot validate itself. Neither can "trade tier A", which was preferred
because tier A came out ahead over the same full sample.

So this simulates someone who does not get to see the answer first. Standing at
date *T*, they know only what happened before *T*. They pick a configuration
from a fixed grid using that history alone, trade it over the window that
follows, and are never allowed to revise it inside that window. Then the clock
moves and they choose again. Stitching those test windows together gives a
record nobody selected for.

Three things the result is compared against, because a number alone means
nothing:

* **the same picks measured in-sample** — the difference is the selection
  premium, the part of the historical figure that was hindsight;
* **the fixed production rule**, held constant across every fold — if adaptive
  selection cannot beat simply never choosing, then choosing is noise;
* **the market in the same window** — a test fold landing in a bull year makes
  every candidate look able.

Purging, and why exit dates rather than an embargo
--------------------------------------------------
A trade entered three days before the test window opens is still running inside
it. Counting it as training data leaks the test period backwards. The usual fix
is a fixed embargo of *horizon* bars; here every trade already carries its own
``exit_date``, so the exact rule is available and is used instead: **a training
trade counts only if it had already exited before the test window opened.**
That is precisely what the analyst standing at *T* would have known, and it
stays correct when candidates have different horizons.

What this still does not fix
----------------------------
The candidate grid was written by someone who has already read the calibration
report. Walk-forward removes the selection bias *inside* each fold; it cannot
remove the bias in deciding which candidates were worth listing at all. That
residue is real and is stated in the report rather than left for the reader to
work out.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import duckdb
import numpy as np
import pandas as pd

from .label import OutcomeRule

#: Below this many trades a window is reported but never selected from, and no
#: conclusion is drawn from it. Same threshold the calibration report uses.
MIN_TRADES_FOR_A_CONCLUSION = 30

ALL_TIERS = ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# selection metrics
# ---------------------------------------------------------------------------

def _mean(returns: np.ndarray, bars: np.ndarray) -> float:
    """Mean return per trade.

    **Not comparable across candidates with different holding periods**, and the
    first run of this module proved it: mean-per-trade selected the 40-day hold
    in all eight folds and then reported it beating the 20-day rule by 3.3
    points, which is very close to what you would get from a coin that holds
    twice as long. Available because it is what the calibration report leads
    with, but it is not the default and the report says so on the page.
    """
    return float(np.mean(returns))


def _median(returns: np.ndarray, bars: np.ndarray) -> float:
    return float(np.median(returns))


def _return_per_bar(returns: np.ndarray, bars: np.ndarray) -> float:
    """Mean return divided by mean bars held: what the capital earns per day.

    The fair comparison when the candidates hold for different lengths of time.
    A 20-day rule returning 3% and a 40-day rule returning 5% are not 5-beats-3;
    the 20-day rule frees the money to do it again, and per bar it is ahead.

    Totals rather than a per-trade ratio: ``mean(return) / mean(bars)`` is the
    return the capital actually earned per day of exposure. Averaging each
    trade's own return-per-bar would instead let a one-bar trade that happened
    to gain 2% dominate a hundred ordinary ones.
    """
    held = float(np.mean(bars))
    if held <= 0:
        return 0.0
    return float(np.mean(returns)) / held


def _return_per_unit_risk(returns: np.ndarray, bars: np.ndarray) -> float:
    """Mean divided by standard deviation of trade returns.

    Not an annualised Sharpe ratio and deliberately not named one — there is no
    risk-free rate here and no time scaling, because the candidates have
    different holding periods and scaling them onto a common axis would invent
    a comparison the data does not support. It is simply mean per unit of
    dispersion, which is what distinguishes a steady small edge from one large
    winner in an otherwise losing set.

    Zero dispersion cannot occur in a window of thirty real trades, but the
    degenerate case still has a right answer and returning 0.0 is not it: a
    constant positive return is the *best* risk-adjusted result there is, and
    scoring it below a lumpy one would rank it backwards. The sign of the mean
    decides.
    """
    spread = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    mean = float(np.mean(returns))
    if spread == 0.0:
        return 0.0 if mean == 0.0 else float(np.inf) * np.sign(mean)
    return mean / spread


SELECTION_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "return-per-bar": _return_per_bar,
    "mean": _mean,
    "median": _median,
    "risk-adjusted": _return_per_unit_risk,
}

#: Comparing candidates that hold for different lengths of time on a per-trade
#: figure hands the contest to whichever holds longest, so the default is the
#: one that does not.
DEFAULT_SELECTION_METRIC = "return-per-bar"


# ---------------------------------------------------------------------------
# candidates and folds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One configuration the analyst could have chosen: an exit rule and which
    tiers to act on.

    Tier membership is a filter applied when reading labels, not a separate
    replay — the tier is a property of the signal, so every tier subset shares
    the same labelled trades.
    """

    rule: OutcomeRule
    tiers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("a candidate must act on at least one tier")
        unknown = set(self.tiers) - set(ALL_TIERS)
        if unknown:
            raise ValueError(f"unknown tier(s): {sorted(unknown)}")

    @property
    def exit_label(self) -> str:
        r = self.rule
        if r.is_hold:
            return f"hold h{r.horizon_bars}"
        target = f"+{r.target_pct:g}" if r.target_pct is not None else "—"
        stop = f"-{r.stop_pct:g}" if r.stop_pct is not None else "—"
        return f"{target}/{stop} h{r.horizon_bars}"

    @property
    def name(self) -> str:
        return f"{self.exit_label} · tier {''.join(self.tiers)}"


def default_grid(horizons: Sequence[int] = (10, 20, 40)) -> list[Candidate]:
    """The grid used unless another is given.

    It re-opens the target/stop question on purpose. The project settled on
    hold-to-horizon and that decision is not being quietly revisited — it is
    being *tested*: if hold really is the better rule, a fold-by-fold contest
    that never sees the future should keep choosing it, and that is a much
    stronger statement than the one-shot comparison which produced it.
    """
    rules = [OutcomeRule(horizon_bars=h) for h in horizons]
    rules += [
        OutcomeRule(horizon_bars=20, target_pct=8.0, stop_pct=4.0),
        OutcomeRule(horizon_bars=20, target_pct=12.0, stop_pct=6.0),
    ]
    tier_sets = [("A",), ("A", "B"), ("A", "B", "C")]
    return [Candidate(rule=r, tiers=t) for r in rules for t in tier_sets]


@dataclass(frozen=True)
class Fold:
    """One train/test split. Test windows are contiguous and never overlap."""

    index: int
    train_start: dt.date
    test_start: dt.date
    test_end: dt.date

    @property
    def label(self) -> str:
        return (
            f"{self.train_start:%Y-%m} → {self.test_start:%Y-%m}  |  "
            f"{self.test_start:%Y-%m-%d} → {self.test_end:%Y-%m-%d}"
        )


def _add_months(date: dt.date, months: int) -> dt.date:
    total = date.month - 1 + months
    year = date.year + total // 12
    month = total % 12 + 1
    day = min(date.day, [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0)
                         else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return dt.date(year, month, day)


def make_folds(
    first: dt.date,
    last: dt.date,
    *,
    test_months: int = 12,
    min_train_months: int = 24,
    train_months: Optional[int] = None,
) -> list[Fold]:
    """Consecutive test windows, each preceded by the history available then.

    ``train_months=None`` gives an **anchored** (expanding) training window —
    every fold trains on everything since the start, which is how a person
    accumulating data actually works. Passing a number gives a **rolling**
    window of that length instead, which asks a different question: whether the
    rule needs recent history specifically, or just a lot of it.
    """
    if last <= first:
        raise ValueError("last must be after first")
    if test_months < 1 or min_train_months < 1:
        raise ValueError("test_months and min_train_months must be positive")
    if train_months is not None and train_months < 1:
        raise ValueError("train_months must be positive when given")

    folds: list[Fold] = []
    test_start = _add_months(first, min_train_months)
    index = 0
    while test_start < last:
        test_end = min(_add_months(test_start, test_months), last)
        # A stub window at the end is worse than no window: it would be reported
        # beside full-length folds as if it carried the same weight.
        if (test_end - test_start).days < 28:
            break
        train_start = (
            first if train_months is None else _add_months(test_start, -train_months)
        )
        if train_start < first:
            train_start = first
        folds.append(
            Fold(index=index, train_start=train_start,
                 test_start=test_start, test_end=test_end)
        )
        index += 1
        test_start = test_end
    return folds


def training_slice(
    labels: pd.DataFrame, fold: Fold, *, purge: bool = True
) -> pd.DataFrame:
    """The trades the analyst standing at ``fold.test_start`` had finished.

    ``purge=False`` exists only so the guard test can show that removing the
    purge changes the answer. Nothing else should ever pass it.
    """
    window = labels[
        (labels["entry_date"] >= fold.train_start)
        & (labels["entry_date"] < fold.test_start)
    ]
    if purge:
        window = window[window["exit_date"] < fold.test_start]
    return window


def testing_slice(labels: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Trades *entered* inside the test window.

    Selected on entry date, not exit date: the decision to enter is the thing
    being judged, and dropping a trade because it happened to run past the
    window's end would discard exactly the entries made late in the window.
    """
    return labels[
        (labels["entry_date"] >= fold.test_start)
        & (labels["entry_date"] <= fold.test_end)
    ]


# ---------------------------------------------------------------------------
# reading labels
# ---------------------------------------------------------------------------

def load_labels(
    con: duckdb.DuckDBPyConnection, rule_version: str
) -> pd.DataFrame:
    """Every resolved, costed trade for one rule, with its signal's tier."""
    frame = con.execute(
        """
        SELECT l.ticker, l.entry_date, l.exit_date, l.return_pct,
               l.bars_held, s.tier
          FROM labels l
          JOIN signals s ON s.ticker = l.ticker AND s.date = l.entry_date
         WHERE l.rule_version = ?
           AND l.return_pct IS NOT NULL
           AND l.exit_date IS NOT NULL
           AND l.bars_held IS NOT NULL
           AND s.tier IS NOT NULL
        """,
        [rule_version],
    ).df()
    for column in ("entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.date
    return frame


def stored_rule_versions(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute("SELECT DISTINCT rule_version FROM labels").fetchall()
    return {r[0] for r in rows}


def universe_return(
    con: duckdb.DuckDBPyConnection,
    *,
    start: dt.date,
    end: dt.date,
    horizon_bars: int,
    buy_fee: float,
    sell_fee: float,
) -> Optional[float]:
    """Mean net return of entering *any* clean bar in the window.

    The honest comparison for a fold: not "did we make money" but "did we make
    more than picking at random over the same stretch of market."
    """
    row = con.execute(
        """
        WITH b AS (
          SELECT p.ticker, p.date, p.open, p.close,
                 row_number() OVER (PARTITION BY p.ticker ORDER BY p.date) AS rn
            FROM prices p
            JOIN price_quality q ON q.ticker = p.ticker AND q.date = p.date
           WHERE q.is_clean
        )
        SELECT avg((x.close * (1 - ?)) / (f.open * (1 + ?)) * 100 - 100)
          FROM b
          JOIN b f ON f.ticker = b.ticker AND f.rn = b.rn + 1
          JOIN b x ON x.ticker = b.ticker AND x.rn = b.rn + ?
         WHERE f.open > 0 AND b.date >= ? AND b.date <= ?
        """,
        [sell_fee, buy_fee, horizon_bars, start, end],
    ).fetchone()
    return None if row is None or row[0] is None else float(row[0])


# ---------------------------------------------------------------------------
# running it
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: Fold
    picked: Optional[Candidate]
    in_sample_score: Optional[float]
    in_sample_mean: Optional[float]
    in_sample_per_bar: Optional[float]
    in_sample_trades: int
    out_of_sample_mean: Optional[float]
    out_of_sample_per_bar: Optional[float]
    out_of_sample_median: Optional[float]
    out_of_sample_win_pct: Optional[float]
    out_of_sample_bars: Optional[float]
    out_of_sample_trades: int
    fixed_mean: Optional[float]
    fixed_per_bar: Optional[float]
    fixed_bars: Optional[float]
    fixed_trades: int
    universe_mean: Optional[float]
    universe_per_bar: Optional[float]
    candidates_considered: int


@dataclass
class WalkForwardResult:
    folds: list[FoldResult] = field(default_factory=list)
    select_by: str = "mean"
    fixed: Optional[Candidate] = None
    anchored: bool = True
    grid_size: int = 0

    @property
    def usable(self) -> list[FoldResult]:
        return [
            f for f in self.folds
            if f.picked is not None
            and f.out_of_sample_trades >= MIN_TRADES_FOR_A_CONCLUSION
        ]

    def pooled(self, attribute: str) -> Optional[float]:
        """Trade-weighted average of a per-fold figure."""
        rows = [
            (getattr(f, attribute), f.out_of_sample_trades)
            for f in self.usable
            if getattr(f, attribute) is not None
        ]
        if not rows:
            return None
        total = sum(n for _, n in rows)
        return sum(v * n for v, n in rows) / total if total else None

    def fold_averaged(self, attribute: str) -> Optional[float]:
        """Unweighted average across folds — every year counts once.

        Reported beside the trade-weighted figure because they answer different
        questions, and where they disagree the disagreement is the finding: it
        means the good years were also the busy ones.
        """
        values = [
            getattr(f, attribute) for f in self.usable
            if getattr(f, attribute) is not None
        ]
        return float(np.mean(values)) if values else None

    @property
    def selection_premium(self) -> Optional[float]:
        """In-sample minus out-of-sample, over the same chosen configurations.

        How many points of the historical result were hindsight rather than
        edge. A large positive number does not mean the strategy fails; it means
        the in-sample figure was never the number to plan around.
        """
        pairs = [
            (f.in_sample_mean, f.out_of_sample_mean)
            for f in self.usable
            if f.in_sample_mean is not None and f.out_of_sample_mean is not None
        ]
        if not pairs:
            return None
        return float(np.mean([a - b for a, b in pairs]))

    @property
    def switches(self) -> tuple[int, int]:
        """``(changes, transitions)`` between consecutive folds.

        Distinct picks alone cannot tell a rule that drifted once from one that
        oscillated every window, and those mean opposite things.
        """
        names = [f.picked.name for f in self.folds if f.picked is not None]
        if len(names) < 2:
            return 0, 0
        changes = sum(1 for a, b in zip(names, names[1:]) if a != b)
        return changes, len(names) - 1

    @property
    def paired_gap(self) -> Optional[tuple[float, float, int]]:
        """Chosen minus fixed, per bar, fold by fold: ``(mean, std error, n)``.

        Comparing the two column averages would answer a weaker question. These
        are paired observations — the same eight test windows, the same market
        — so the difference is taken inside each fold and only then averaged,
        and it carries its own spread.

        Eight folds is a small, non-independent sample and the standard error
        here is a rough guide, not a p-value. It is enough for the one call the
        report has to make: whether a gap is bigger than the fold-to-fold noise
        it sits in.
        """
        pairs = [
            f.out_of_sample_per_bar - f.fixed_per_bar
            for f in self.usable
            if f.out_of_sample_per_bar is not None and f.fixed_per_bar is not None
        ]
        if len(pairs) < 2:
            return None
        mean = float(np.mean(pairs))
        error = float(np.std(pairs, ddof=1)) / float(np.sqrt(len(pairs)))
        return mean, error, len(pairs)

    @property
    def distinct_picks(self) -> list[str]:
        seen: list[str] = []
        for f in self.folds:
            if f.picked is not None and f.picked.name not in seen:
                seen.append(f.picked.name)
        return seen


def run(
    con: duckdb.DuckDBPyConnection,
    candidates: Sequence[Candidate],
    folds: Sequence[Fold],
    *,
    labels_by_rule: dict[str, pd.DataFrame],
    select_by: str = DEFAULT_SELECTION_METRIC,
    fixed: Optional[Candidate] = None,
    buy_fee: float = 0.0,
    sell_fee: float = 0.0,
    anchored: bool = True,
) -> WalkForwardResult:
    """Choose on the training window, measure on the test window, never both."""
    if select_by not in SELECTION_METRICS:
        raise ValueError(
            f"unknown selection metric {select_by!r}; "
            f"choose from {sorted(SELECTION_METRICS)}"
        )
    missing = {c.rule.version for c in candidates} - set(labels_by_rule)
    if fixed is not None and fixed.rule.version not in labels_by_rule:
        missing.add(fixed.rule.version)
    if missing:
        raise ValueError(
            "no labels supplied for rule(s) " + ", ".join(sorted(missing))
            + ". A candidate silently scoring zero trades would drop out of "
            "every fold and look like it was simply never the best."
        )

    score = SELECTION_METRICS[select_by]
    result = WalkForwardResult(
        select_by=select_by, fixed=fixed, anchored=anchored, grid_size=len(candidates)
    )

    def sample_for(
        candidate: Candidate, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        subset = frame[frame["tier"].isin(candidate.tiers)]
        return (
            subset["return_pct"].to_numpy(dtype=float),
            subset["bars_held"].to_numpy(dtype=float),
        )

    def per_bar(returns: np.ndarray, bars: np.ndarray) -> Optional[float]:
        if not len(returns):
            return None
        held = float(np.mean(bars))
        return None if held <= 0 else float(np.mean(returns)) / held

    for fold in folds:
        best: Optional[Candidate] = None
        best_score: Optional[float] = None
        best_mean: Optional[float] = None
        best_per_bar: Optional[float] = None
        best_trades = 0
        considered = 0

        for candidate in candidates:
            labels = labels_by_rule[candidate.rule.version]
            returns, bars = sample_for(candidate, training_slice(labels, fold))
            if len(returns) < MIN_TRADES_FOR_A_CONCLUSION:
                continue
            considered += 1
            value = score(returns, bars)
            if best_score is None or value > best_score:
                best, best_score = candidate, value
                best_mean = float(np.mean(returns))
                best_per_bar = per_bar(returns, bars)
                best_trades = len(returns)

        oos_mean = oos_median = oos_win = oos_per_bar = oos_bars = None
        oos_trades = 0
        universe = universe_per_bar = None
        if best is not None:
            returns, bars = sample_for(
                best, testing_slice(labels_by_rule[best.rule.version], fold)
            )
            oos_trades = len(returns)
            if oos_trades:
                oos_mean = float(np.mean(returns))
                oos_median = float(np.median(returns))
                oos_win = float(100.0 * np.mean(returns > 0))
                oos_bars = float(np.mean(bars))
                oos_per_bar = per_bar(returns, bars)
            universe = universe_return(
                con,
                start=fold.test_start,
                end=fold.test_end,
                horizon_bars=best.rule.horizon_bars,
                buy_fee=buy_fee,
                sell_fee=sell_fee,
            )
            if universe is not None:
                # The baseline holds exactly the horizon, by construction.
                universe_per_bar = universe / best.rule.horizon_bars

        fixed_mean = fixed_per_bar = fixed_bars = None
        fixed_trades = 0
        if fixed is not None:
            returns, bars = sample_for(
                fixed, testing_slice(labels_by_rule[fixed.rule.version], fold)
            )
            fixed_trades = len(returns)
            if fixed_trades:
                fixed_mean = float(np.mean(returns))
                fixed_bars = float(np.mean(bars))
                fixed_per_bar = per_bar(returns, bars)

        result.folds.append(
            FoldResult(
                fold=fold,
                picked=best,
                in_sample_score=best_score,
                in_sample_mean=best_mean,
                in_sample_per_bar=best_per_bar,
                in_sample_trades=best_trades,
                out_of_sample_mean=oos_mean,
                out_of_sample_per_bar=oos_per_bar,
                out_of_sample_median=oos_median,
                out_of_sample_win_pct=oos_win,
                out_of_sample_bars=oos_bars,
                out_of_sample_trades=oos_trades,
                fixed_mean=fixed_mean,
                fixed_per_bar=fixed_per_bar,
                fixed_bars=fixed_bars,
                fixed_trades=fixed_trades,
                universe_mean=universe,
                universe_per_bar=universe_per_bar,
                candidates_considered=considered,
            )
        )
    return result


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _fmt(value, spec: str = "+.2f") -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return format(value, spec)


def render_markdown(
    result: WalkForwardResult,
    *,
    cost_description: str,
    generated_at: Optional[dt.datetime] = None,
) -> str:
    out: list[str] = []
    add = out.append
    now = generated_at or dt.datetime.now()

    add("# Walk-forward validation")
    add("")
    add(f"Generated {now:%Y-%m-%d %H:%M}. Reproducible from the DuckDB store by "
        "re-running the command.")
    add("")
    add("```")
    add(f"grid            : {result.grid_size} configurations "
        f"(exit rule × tier subset)")
    add(f"training window : {'anchored (expanding)' if result.anchored else 'rolling'}")
    add(f"chosen by       : {result.select_by} net return over the training window")
    add(f"fixed comparison: {result.fixed.name if result.fixed else 'none'}")
    add("")
    add(cost_description)
    add("```")
    add("")
    add("Each fold picks one configuration using **only trades that had already "
        "exited** before the test window opened, then trades that choice through "
        "the window without revision.")
    add("")

    add("## Fold by fold")
    add("")
    add("Per-trade returns are shown for readability, but the comparison that "
        "counts is **per bar held** — a rule that holds twice as long earns "
        "roughly twice as much per trade without being any better.")
    add("")
    add("| Fold | Test window | Chosen on the training data | Bars held | "
        "In-sample % | Out-of-sample % | Per bar | Market, same horizon | Trades |")
    add("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for f in result.folds:
        if f.picked is None:
            add(f"| {f.fold.index + 1} | {f.fold.test_start:%Y-%m-%d} → "
                f"{f.fold.test_end:%Y-%m-%d} | *no candidate had enough trades* "
                "| — | — | — | — | — | — |")
            continue
        flag = "" if f.out_of_sample_trades >= MIN_TRADES_FOR_A_CONCLUSION else " ⚠️"
        add(
            f"| {f.fold.index + 1}{flag} | {f.fold.test_start:%Y-%m-%d} → "
            f"{f.fold.test_end:%Y-%m-%d} | {f.picked.name} | "
            f"{_fmt(f.out_of_sample_bars, '.0f')} | "
            f"{_fmt(f.in_sample_mean)} | {_fmt(f.out_of_sample_mean)} | "
            f"{_fmt(f.out_of_sample_per_bar, '+.3f')} | "
            f"{_fmt(f.universe_mean)} | {f.out_of_sample_trades:,} |"
        )
    add("")

    thin = [f for f in result.folds
            if f.picked is not None
            and f.out_of_sample_trades < MIN_TRADES_FOR_A_CONCLUSION]
    if thin:
        add(f"⚠️ Folds with fewer than {MIN_TRADES_FOR_A_CONCLUSION} out-of-sample "
            "trades are shown but excluded from every aggregate below.")
        add("")

    add("## What the choosing was worth")
    add("")
    pooled = result.pooled("out_of_sample_mean")
    averaged = result.fold_averaged("out_of_sample_mean")
    premium = result.selection_premium
    market = result.fold_averaged("universe_mean")
    fixed_mean = result.fold_averaged("fixed_mean")
    chosen_bar = result.fold_averaged("out_of_sample_per_bar")
    fixed_bar = result.fold_averaged("fixed_per_bar")
    market_bar = result.fold_averaged("universe_per_bar")
    chosen_bars = result.fold_averaged("out_of_sample_bars")
    fixed_bars = result.fold_averaged("fixed_bars")

    add("| | Bars held | Per trade % | **Per bar %** |")
    add("|---|---:|---:|---:|")
    add(f"| Chosen each fold, out-of-sample | {_fmt(chosen_bars, '.0f')} | "
        f"{_fmt(averaged)} | **{_fmt(chosen_bar, '+.3f')}** |")
    if fixed_mean is not None:
        add(f"| Fixed rule ({result.fixed.name}) | {_fmt(fixed_bars, '.0f')} | "
            f"{_fmt(fixed_mean)} | **{_fmt(fixed_bar, '+.3f')}** |")
    add(f"| Market, each fold's chosen horizon | {_fmt(chosen_bars, '.0f')} | "
        f"{_fmt(market)} | **{_fmt(market_bar, '+.3f')}** |")
    add("")
    add(f"One vote per fold. Trade-weighted, the out-of-sample per-trade figure "
        f"is {_fmt(pooled)}% — higher, because the busiest folds were also the "
        "best ones, which is itself worth knowing.")
    add("")

    if premium is not None:
        add(f"**Selection premium: {_fmt(premium)} points per trade.** How much "
            "better the chosen configuration looked on the data used to choose "
            "it than it went on to perform. A positive number is the part of a "
            "backtested figure that hindsight paid for. A negative one does not "
            "mean the method is conservative — it means the test windows "
            "happened to be kinder than the training windows, which is a fact "
            "about this decade, not about the rule.")
        add("")

    if result.fixed is not None:
        same = all(
            f.picked is not None and f.picked.name == result.fixed.name
            for f in result.usable
        ) and bool(result.usable)
        paired = result.paired_gap

        if same:
            add(f"**Choosing and not choosing are the same thing here.** All "
                f"{len(result.usable)} folds independently arrived at the fixed "
                "rule, so there is no gap to report. That is the strongest "
                "result the fixed rule could have got: it was re-derived from "
                "data that had not seen its future.")
        elif paired is None:
            add("Not enough paired folds to say whether choosing was worth "
                "anything.")
        else:
            gap, error, n = paired
            add(f"**Was choosing worth anything?** Fold by fold, re-picking came "
                f"out {_fmt(gap, '+.3f')} points per bar against the fixed rule, "
                f"with a standard error of {error:.3f} across {n} folds.")
            add("")
            if abs(gap) <= error:
                add("**That is indistinguishable from zero.** The selection "
                    "machinery, run honestly, bought nothing the fixed rule did "
                    "not already have. Read the right way round, this is the "
                    "reassuring result: the rule already decided is as good as "
                    "anything this grid could find, and it does not need a "
                    "tuning step bolted onto it.")
            elif gap > 0:
                add("**That is larger than the fold-to-fold noise**, so the "
                    "selection step is doing something — on eight overlapping "
                    "years of one exchange, which is weak evidence for a "
                    "positive claim even when it points the right way.")
            else:
                add("**That is larger than the fold-to-fold noise, and it is "
                    "negative.** Re-picking each fold actively lost to holding "
                    "the fixed rule throughout: the selection step is fitting "
                    "noise, and the simpler answer is to stop selecting.")
        add("")

    if market_bar is not None and chosen_bar is not None:
        beaten = sum(
            1 for f in result.usable
            if f.out_of_sample_per_bar is not None and f.universe_per_bar is not None
            and f.out_of_sample_per_bar > f.universe_per_bar
        )
        add(f"Against the market: the out-of-sample record beat entering at "
            f"random over the same horizon in **{beaten} of "
            f"{len(result.usable)}** folds.")
        add("")

    add("## Did it keep choosing the same thing?")
    add("")
    picks = result.distinct_picks
    if len(picks) <= 1:
        add(f"Every fold chose the same configuration: **{picks[0] if picks else '—'}**. "
            "A stable choice is the good outcome — it means the training windows "
            "agreed, rather than each one finding a different favourite.")
    else:
        add(f"{len(picks)} different configurations were chosen across "
            f"{len(result.folds)} folds:")
        add("")
        for name in picks:
            folds_with = [str(f.fold.index + 1) for f in result.folds
                          if f.picked is not None and f.picked.name == name]
            add(f"- **{name}** — fold(s) {', '.join(folds_with)}")
        add("")
        changes, transitions = result.switches
        add(f"It changed **{changes} time(s) in {transitions}** moves from one "
            "fold to the next.")
        add("")
        if transitions and changes / transitions > 0.5:
            add("That is a choice being pushed around by whatever the last "
                "window happened to contain, not a strategy being tuned. A "
                "selection step this unstable is a source of variance, not of "
                "edge.")
        else:
            add("That is drift rather than oscillation: the training windows "
                "mostly agreed and the answer moved a few times as history "
                "accumulated. Worth reading as a change in the market rather "
                "than a failure of the method — but only worth acting on if "
                "the gap against the fixed rule above is real, and it is not.")
    add("")

    add("## Read this before using any number above")
    add("")
    add("- **The grid was written by someone who had already read the "
        "calibration report.** Walk-forward removes the selection bias inside "
        "each fold. It cannot remove the bias in deciding which configurations "
        "were worth listing, and no amount of folding will.")
    add("- **Folds are not independent.** They are consecutive slices of one "
        "market's history, and a regime spans several of them. Eight folds are "
        "not eight independent experiments.")
    add("- **Fees may be assumed.** Check the cost line at the top. If it says "
        "ASSUMED, every return here inherits that assumption.")
    add("- **The final fold is short of its late entries.** A trade needs its "
        "whole horizon inside the stored data to resolve, so entries in the "
        "last weeks of the newest fold have no outcome yet and are absent. "
        "That truncation bites the longer holds hardest, which is exactly "
        "where the comparison is most delicate.")
    add("- **Per-trade and per-bar can disagree, and per bar is the one to "
        "use.** Selecting on per-trade return hands the contest to whichever "
        "candidate holds longest; that is a property of the metric, not a "
        "finding about the rule.")
    add("- **Out-of-sample is not out-of-market.** Every fold is IDX equities "
        "over the same decade. Nothing here says the rule survives a different "
        "exchange, or the next decade of this one.")
    add("")
    return "\n".join(out)


def write(path: str | Path, markdown: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(markdown, encoding="utf-8")
    return p
