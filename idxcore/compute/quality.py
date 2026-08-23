"""Data-quality gates (IVI-73).

Bad data that looks like good data is the main risk in this project. A stale
price makes RSI sit at 50 and volatility collapse; an unrecorded split makes a
1:10 look like a genuine -90% day and poisons every moving average that touches
it for the next 200 bars.

Every check here **flags** the row. None of them delete it:

* deleting a bar silently rewrites history, and
* a gap we created ourselves is indistinguishable from one the exchange had.

Downstream code decides what to exclude; this module only decides what is true.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

#: A close-to-close move beyond this, with no split recorded, is almost always
#: an unadjusted corporate action rather than a real trade.
JUMP_THRESHOLD_PCT = 35.0

#: Consecutive identical closes at or above this length count as stale. Common
#: on illiquid IDX small caps, where the same price prints for weeks.
STALE_RUN_LENGTH = 5

#: The deepest MA the signal engine can ask for. Fewer bars than this means
#: MA200 does not exist — which is its own state, not a failed test.
DEEPEST_MA = 200

#: More than this many calendar days between bars is a gap worth flagging.
#: IDX closes for Idul Fitri for about a week, so the threshold sits above a
#: normal long weekend plus a public holiday.
GAP_DAYS = 10

QUALITY_COLUMNS = [
    "ticker",
    "date",
    "insufficient_history",
    "stale_price",
    "zero_volume",
    "suspicious_jump",
    "gap",
    "impossible_bar",
    "stale_run_length",
    "jump_pct",
    "gap_days",
    "bars_available",
    "is_clean",
    "computed_at",
]


@dataclass
class QualitySummary:
    tickers: int = 0
    rows: int = 0
    insufficient_history: int = 0
    stale_price: int = 0
    zero_volume: int = 0
    suspicious_jump: int = 0
    gap: int = 0
    impossible_bar: int = 0
    clean: int = 0

    def render(self) -> str:
        def pct(n: int) -> str:
            return f"{(n / self.rows * 100):5.2f}%" if self.rows else "    -"

        return "\n".join(
            [
                "quality gates",
                f"  tickers checked      : {self.tickers}",
                f"  bars checked         : {self.rows}",
                f"  clean                : {self.clean} ({pct(self.clean)})",
                "  ---",
                f"  insufficient_history : {self.insufficient_history} ({pct(self.insufficient_history)})",
                f"  stale_price          : {self.stale_price} ({pct(self.stale_price)})",
                f"  zero_volume          : {self.zero_volume} ({pct(self.zero_volume)})",
                f"  suspicious_jump      : {self.suspicious_jump} ({pct(self.suspicious_jump)})",
                f"  gap                  : {self.gap} ({pct(self.gap)})",
                f"  impossible_bar       : {self.impossible_bar} ({pct(self.impossible_bar)})",
            ]
        )


def stale_run_lengths(close: pd.Series) -> pd.Series:
    """Length of the run of identical closes ending at each bar.

    A bar whose close differs from the previous one starts a new run of 1.
    """
    close = close.reset_index(drop=True)
    changed = close.ne(close.shift())
    group = changed.cumsum()
    return close.groupby(group).cumcount().add(1)


def flag_ticker(
    frame: pd.DataFrame,
    *,
    jump_threshold: float = JUMP_THRESHOLD_PCT,
    stale_run: int = STALE_RUN_LENGTH,
    deepest_ma: int = DEEPEST_MA,
    gap_days: int = GAP_DAYS,
    computed_at: Optional[dt.datetime] = None,
) -> pd.DataFrame:
    """Compute the quality flag set for one ticker's price history.

    `frame` must be a single ticker's bars, ascending by date, with the
    columns of the `prices` table.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=QUALITY_COLUMNS)

    data = frame.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(data["close"], errors="coerce")

    out = pd.DataFrame(
        {
            "ticker": data["ticker"],
            "date": data["date"],
        }
    )

    # bars_available counts bars up to and including this one, so a row can be
    # judged on what was knowable at the time rather than on the full series.
    out["bars_available"] = np.arange(1, len(data) + 1)
    out["insufficient_history"] = out["bars_available"] < deepest_ma

    runs = stale_run_lengths(close)
    out["stale_run_length"] = runs.astype("Int64")
    out["stale_price"] = runs >= stale_run

    volume = pd.to_numeric(data["volume"], errors="coerce")
    out["zero_volume"] = volume.fillna(0) <= 0

    # A split is a multiplier, so a recorded split explains the jump. Yahoo
    # reports 1.0 for "no split"; anything else means the action is known.
    split = pd.to_numeric(data["split_factor"], errors="coerce").fillna(1.0)
    has_split = split.ne(1.0)

    pct_change = close.pct_change() * 100.0
    out["jump_pct"] = pct_change
    out["suspicious_jump"] = pct_change.abs().gt(jump_threshold) & ~has_split

    dates = pd.to_datetime(data["date"])
    delta = dates.diff().dt.days
    out["gap_days"] = delta.astype("Int64")
    out["gap"] = delta.gt(gap_days).fillna(False)

    # An OHLC bar where the high is not the highest of the four, or the low not
    # the lowest, is arithmetically impossible — corruption in the feed, not a
    # real session. `verify` already reports these; flagging them here is what
    # keeps them out of the signals and the backtest. NaN comparisons are False,
    # so a padded/empty bar is never mistaken for a corrupt one.
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    open_ = pd.to_numeric(data["open"], errors="coerce")
    out["impossible_bar"] = (
        (high < low)
        | (high < open_) | (high < close)
        | (low > open_) | (low > close)
    ).fillna(False)

    for column in (
        "insufficient_history", "stale_price", "zero_volume",
        "suspicious_jump", "gap", "impossible_bar",
    ):
        out[column] = out[column].fillna(False).astype(bool)

    # is_clean deliberately excludes insufficient_history: a young listing is
    # not damaged data, it simply has not lived long enough for MA200. Treating
    # the two the same is the silent-downgrade bug in another costume.
    out["is_clean"] = ~(
        out["stale_price"] | out["zero_volume"] | out["suspicious_jump"]
        | out["gap"] | out["impossible_bar"]
    )
    out["computed_at"] = computed_at or dt.datetime.now()

    return out.loc[:, QUALITY_COLUMNS]


def flag_all(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: Optional[list[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Flag every stored bar, one ticker at a time."""
    if tickers is None:
        rows = con.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchall()
        tickers = [r[0] for r in rows]

    frames = []
    for ticker in tickers:
        prices = con.execute(
            "SELECT * FROM prices WHERE ticker = ? ORDER BY date", [ticker]
        ).df()
        if prices.empty:
            continue
        frames.append(flag_ticker(prices, **kwargs))

    if not frames:
        return pd.DataFrame(columns=QUALITY_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def upsert_quality(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> int:
    if frame is None or frame.empty:
        return 0
    payload = frame.loc[:, QUALITY_COLUMNS].drop_duplicates(
        subset=["ticker", "date"], keep="last"
    )
    con.register("_incoming_quality", payload)
    try:
        con.execute(
            """
            INSERT INTO price_quality
                (ticker, date, insufficient_history, stale_price, zero_volume,
                 suspicious_jump, gap, impossible_bar, stale_run_length, jump_pct,
                 gap_days, bars_available, is_clean, computed_at)
            SELECT ticker, date, insufficient_history, stale_price, zero_volume,
                   suspicious_jump, gap, impossible_bar, stale_run_length, jump_pct,
                   gap_days, bars_available, is_clean, computed_at
            FROM _incoming_quality
            ON CONFLICT (ticker, date) DO UPDATE SET
                insufficient_history = excluded.insufficient_history,
                stale_price          = excluded.stale_price,
                zero_volume          = excluded.zero_volume,
                suspicious_jump      = excluded.suspicious_jump,
                gap                  = excluded.gap,
                impossible_bar       = excluded.impossible_bar,
                stale_run_length     = excluded.stale_run_length,
                jump_pct             = excluded.jump_pct,
                gap_days             = excluded.gap_days,
                bars_available       = excluded.bars_available,
                is_clean             = excluded.is_clean,
                computed_at          = excluded.computed_at
            """
        )
    finally:
        con.unregister("_incoming_quality")
    return len(payload)


def summarise(con: duckdb.DuckDBPyConnection) -> QualitySummary:
    """Counts per flag, so the scale of each problem is visible."""
    row = con.execute(
        """
        SELECT count(DISTINCT ticker),
               count(*),
               count(*) FILTER (WHERE insufficient_history),
               count(*) FILTER (WHERE stale_price),
               count(*) FILTER (WHERE zero_volume),
               count(*) FILTER (WHERE suspicious_jump),
               count(*) FILTER (WHERE gap),
               count(*) FILTER (WHERE impossible_bar),
               count(*) FILTER (WHERE is_clean)
        FROM price_quality
        """
    ).fetchone()
    return QualitySummary(*row)
