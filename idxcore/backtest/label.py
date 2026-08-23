"""Outcome labelling (IVI-79).

"Did this signal work?" needs a definition before anything can be measured.
Without one, every probability the dashboard prints is decoration.

What is decided here, and what is not
-------------------------------------
The **target, stop and horizon are the owner's decision** and are not guessed at
in code. :class:`OutcomeRule` carries them, and the CLI requires them to be
stated explicitly. There is no default that quietly becomes the answer.

The **tie-break is decided**: if a bar's high touches the target and its low
touches the stop on the same day, it counts as a **stop**. Daily bars do not
record the order the two were hit, and assuming the favourable one is the single
most common way a backtest lies to itself. The brief calls for the unfavourable
resolution and that is what ``TIE_BREAK_UNFAVOURABLE`` means.

Lookahead
---------
Labelling reads only bars **strictly after** the signal date, and the entry
fills at the **next** bar's open. You cannot buy at a close you only learned
about after the close.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .costs import CostModel

TIE_BREAK_UNFAVOURABLE = "unfavourable"
TIE_BREAK_FAVOURABLE = "favourable"

OUTCOME_TARGET = "target"
OUTCOME_STOP = "stop"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_OPEN = "open"

LABEL_COLUMNS = [
    "ticker",
    "entry_date",
    "rule_version",
    "fill_date",
    "fill_price",
    "exit_date",
    "exit_price",
    "outcome",
    "bars_held",
    "return_pct",
    "gross_return_pct",
    "mae",
    "mfe",
    "computed_at",
]


@dataclass(frozen=True)
class OutcomeRule:
    """The definition of a win. Every field must be chosen deliberately.

    `target_pct` and `stop_pct` are positive percentages away from the fill:
    target 8.0 means +8%, stop 4.0 means -4%. **Either may be None**, meaning
    that leg simply does not exist.

    With both None the rule is *hold to horizon*: enter, wait the full window,
    exit at the close. That was the decision taken for this project on
    2026-08-21, and it was taken on evidence rather than taste — a -4% stop sat
    inside the median adverse excursion (-4.44%), so it was triggered by
    ordinary noise 61% of the time, while a target capped exactly the large
    winners that carry a fat-tailed return distribution.

    A hold rule has no downside protection. That is a real property of the
    choice, not an oversight: MAE is still recorded on every trade so the
    drawdown you would have sat through stays visible.
    """

    horizon_bars: int
    target_pct: Optional[float] = None
    stop_pct: Optional[float] = None
    tie_break: str = TIE_BREAK_UNFAVOURABLE
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("target_pct", "stop_pct"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} is a positive distance from the fill, or None")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be at least 1")
        if self.tie_break not in (TIE_BREAK_UNFAVOURABLE, TIE_BREAK_FAVOURABLE):
            raise ValueError(f"unknown tie_break {self.tie_break!r}")

    @property
    def is_hold(self) -> bool:
        """No target and no stop: ride the position to the horizon."""
        return self.target_pct is None and self.stop_pct is None

    @property
    def version(self) -> str:
        """Stable identity for this rule, used as part of the labels key.

        Re-running under a changed definition writes new rows rather than
        overwriting the previous measurement, so two rules can be compared
        instead of one silently replacing the other.
        """
        payload = json.dumps(
            {k: v for k, v in asdict(self).items() if k != "label"}, sort_keys=True
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
        if self.label:
            stem = self.label
        elif self.is_hold:
            stem = f"hold-h{self.horizon_bars}"
        else:
            stem = (
                f"t{self.target_pct if self.target_pct is not None else 'x'}"
                f"-s{self.stop_pct if self.stop_pct is not None else 'x'}"
                f"-h{self.horizon_bars}-{self.tie_break[:5]}"
            )
        return f"{stem}-{digest}"

    def describe(self) -> str:
        lines = [f"outcome rule {self.version}"]
        if self.is_hold:
            lines.append(f"  hold     : {self.horizon_bars} trading days, exit at close")
            lines.append("  target   : none")
            lines.append("  stop     : none - no downside protection, MAE still recorded")
            return "\n".join(lines)

        lines.append(
            f"  target   : +{self.target_pct:g}%" if self.target_pct else "  target   : none"
        )
        lines.append(
            f"  stop     : -{self.stop_pct:g}%" if self.stop_pct else "  stop     : none"
        )
        lines.append(f"  horizon  : {self.horizon_bars} trading days")
        if self.target_pct and self.stop_pct:
            lines.append(
                f"  tie-break: {self.tie_break} "
                f"({'stop wins' if self.tie_break == TIE_BREAK_UNFAVOURABLE else 'TARGET WINS - optimistic'})"
            )
        return "\n".join(lines)


#: The rule this project actually trades, decided in IVI-79: hold to horizon,
#: 20 trading days, no target and no stop.
#:
#: Named here rather than inferred anywhere, because `labels` holds more than
#: one rule and always will — the backtest keys on `rule_version` precisely so
#: that a new measurement adds a row instead of overwriting the last one, and
#: walk-forward validation deliberately labels candidates nobody intends to
#: trade. "Whichever was computed most recently" is therefore not a synonym for
#: "the one we act on", and reading it as one put a rejected target/stop rule's
#: figures on the dashboard under the heading of measured results.
DECIDED_RULE = OutcomeRule(horizon_bars=20)


@dataclass
class Trade:
    ticker: str
    entry_date: dt.date
    fill_date: Optional[dt.date] = None
    fill_price: Optional[float] = None
    exit_date: Optional[dt.date] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None
    bars_held: Optional[int] = None
    gross_return_pct: Optional[float] = None
    return_pct: Optional[float] = None
    mae: Optional[float] = None
    mfe: Optional[float] = None


def label_one(
    bars: pd.DataFrame,
    entry_index: int,
    rule: OutcomeRule,
    *,
    ticker: str,
    costs: Optional[CostModel] = None,
) -> Optional[Trade]:
    """Label a single signal.

    `bars` is one ticker's OHLC ascending by date; `entry_index` is the row of
    the signal. Everything read is strictly after that row.

    Returns None when there is no next bar to fill on — a signal on the last
    stored day has no measurable outcome, and inventing one would be lookahead.
    """
    fill_index = entry_index + 1
    if fill_index >= len(bars):
        return None

    entry_date = bars["date"].iloc[entry_index]
    fill_row = bars.iloc[fill_index]
    fill_price = float(fill_row["open"])
    if not np.isfinite(fill_price) or fill_price <= 0:
        return None

    effective_fill = costs.buy_price(fill_price) if costs else fill_price

    trade = Trade(
        ticker=ticker,
        entry_date=entry_date,
        fill_date=fill_row["date"],
        fill_price=effective_fill,
    )

    target_price = (
        effective_fill * (1.0 + rule.target_pct / 100.0)
        if rule.target_pct is not None
        else None
    )
    stop_price = (
        effective_fill * (1.0 - rule.stop_pct / 100.0)
        if rule.stop_pct is not None
        else None
    )

    # The holding window starts at the fill bar itself: a gap through the stop
    # on the day we bought is a real loss, not something to skip.
    last_index = min(fill_index + rule.horizon_bars - 1, len(bars) - 1)
    window = bars.iloc[fill_index : last_index + 1]

    mae = 0.0
    mfe = 0.0
    for offset, (_, bar) in enumerate(window.iterrows()):
        high = float(bar["high"])
        low = float(bar["low"])
        if np.isfinite(low):
            mae = min(mae, (low / effective_fill - 1.0) * 100.0)
        if np.isfinite(high):
            mfe = max(mfe, (high / effective_fill - 1.0) * 100.0)

        hit_target = (
            target_price is not None and np.isfinite(high) and high >= target_price
        )
        hit_stop = stop_price is not None and np.isfinite(low) and low <= stop_price

        if hit_target and hit_stop:
            # Same bar touched both. A daily bar does not record which came
            # first, so resolve against ourselves.
            resolved = (
                OUTCOME_STOP
                if rule.tie_break == TIE_BREAK_UNFAVOURABLE
                else OUTCOME_TARGET
            )
        elif hit_target:
            resolved = OUTCOME_TARGET
        elif hit_stop:
            resolved = OUTCOME_STOP
        else:
            continue

        exit_price = target_price if resolved == OUTCOME_TARGET else stop_price
        assert exit_price is not None
        _close_trade(trade, bar, resolved, exit_price, offset, costs)
        trade.mae, trade.mfe = mae, mfe
        return trade

    # Never resolved inside the horizon.
    final = window.iloc[-1]
    reached_horizon = len(window) >= rule.horizon_bars
    outcome = OUTCOME_TIMEOUT if reached_horizon else OUTCOME_OPEN
    _close_trade(
        trade, final, outcome, float(final["close"]), len(window) - 1, costs
    )
    trade.mae, trade.mfe = mae, mfe
    if outcome == OUTCOME_OPEN:
        # Still running when the data ends. Recorded, but it is not a result:
        # excluding it would bias the sample, and counting it as a timeout
        # would claim an exit that never happened.
        trade.return_pct = None
        trade.gross_return_pct = None
    return trade


def _close_trade(
    trade: Trade,
    bar: pd.Series,
    outcome: str,
    exit_price: float,
    bars_held: int,
    costs: Optional[CostModel],
) -> None:
    trade.exit_date = bar["date"]
    trade.outcome = outcome
    trade.bars_held = int(bars_held)
    fill = float(trade.fill_price)

    if costs is not None:
        realised_exit = costs.sell_price(exit_price)
        trade.exit_price = realised_exit
        trade.gross_return_pct = (realised_exit / fill - 1.0) * 100.0
        trade.return_pct = costs.net_return_pct(fill, exit_price)
    else:
        trade.exit_price = exit_price
        trade.gross_return_pct = (exit_price / fill - 1.0) * 100.0
        # Left NULL deliberately: return_pct means "net of costs", and with no
        # cost model there is no honest value to put here.
        trade.return_pct = None


def label_frame(
    trades: Iterable[Trade],
    rule: OutcomeRule,
    *,
    computed_at: Optional[dt.datetime] = None,
) -> pd.DataFrame:
    now = computed_at or dt.datetime.now()
    rows = []
    for t in trades:
        rows.append(
            {
                "ticker": t.ticker,
                "entry_date": t.entry_date,
                "rule_version": rule.version,
                "fill_date": t.fill_date,
                "fill_price": t.fill_price,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "outcome": t.outcome,
                "bars_held": t.bars_held,
                "return_pct": t.return_pct,
                "gross_return_pct": t.gross_return_pct,
                "mae": t.mae,
                "mfe": t.mfe,
                "computed_at": now,
            }
        )
    if not rows:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    return pd.DataFrame(rows).loc[:, LABEL_COLUMNS]


def upsert_labels(con, frame: pd.DataFrame) -> int:
    if frame is None or frame.empty:
        return 0
    payload = frame.loc[:, LABEL_COLUMNS].drop_duplicates(
        subset=["ticker", "entry_date", "rule_version"], keep="last"
    )
    con.register("_incoming_labels", payload)
    try:
        con.execute(
            """
            INSERT INTO labels
                (ticker, entry_date, rule_version, fill_date, fill_price,
                 exit_date, exit_price, outcome, bars_held, return_pct,
                 gross_return_pct, mae, mfe, computed_at)
            SELECT ticker, entry_date, rule_version, fill_date, fill_price,
                   exit_date, exit_price, outcome, bars_held, return_pct,
                   gross_return_pct, mae, mfe, computed_at
            FROM _incoming_labels
            ON CONFLICT (ticker, entry_date, rule_version) DO UPDATE SET
                fill_date        = excluded.fill_date,
                fill_price       = excluded.fill_price,
                exit_date        = excluded.exit_date,
                exit_price       = excluded.exit_price,
                outcome          = excluded.outcome,
                bars_held        = excluded.bars_held,
                return_pct       = excluded.return_pct,
                gross_return_pct = excluded.gross_return_pct,
                mae              = excluded.mae,
                mfe              = excluded.mfe,
                computed_at      = excluded.computed_at
            """
        )
    finally:
        con.unregister("_incoming_labels")
    return len(payload)
