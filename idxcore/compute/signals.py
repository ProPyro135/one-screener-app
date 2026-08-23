"""Signal engine (IVI-75, IVI-76).

The six criteria are not six rules. Run the draft's logic on a perfect setup and
all six fire at once::

    Perfect full uptrend, RSI 60  ->  ['K4','K1','K5','K2','K6','K3']

They are nested supersets of **two axes**:

* ``alignment_depth`` in {0, 3, 4, 5} — how deep the MA stack is ordered
* ``rsi_gate`` — whether RSI(14) > 50

Those two are the stored truth. The six K labels are a *rendering* of them,
produced by :func:`render_criteria`, and are never stored as six flags.

Two defects this module exists to fix
-------------------------------------
**Missing MAs must not read as a passing signal.** A ticker with 60 bars has no
MA100 or MA200, and the draft labelled it identically to a genuine shallow
alignment::

    MA100/MA200 = NaN, perfect 5>20>50, RSI 60  ->  ['K6','K3']

Byte-identical to a real depth-3 stock. Here, a stock with fewer than 200 bars
can never reach depth 5, and the row carries
``data_quality = 'insufficient_history'`` so the two stay distinguishable.

**The MA5 filter excluded the best setups.** The draft required a cross *today*
(``close > ma5 AND close_prev <= ma5_prev``), so a stock in full 5>20>50>100>200
alignment that had held above MA5 for three weeks returned nothing. The
condition is now simply ``close > ma5``, with ``days_since_ma5_cross`` recorded
alongside it so trend maturity is visible rather than binary.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

#: Bars needed before the corresponding depth can even be evaluated.
DEPTH_REQUIREMENTS = {3: 50, 4: 100, 5: 200}

RSI_GATE_LEVEL = 50.0

#: Fallback contamination window when no depth is claimed - MA5 still feeds
#: above_ma5, so a bad bar matters for at least this long.
MA5_WINDOW = 5

SIGNAL_COLUMNS = [
    "ticker",
    "date",
    "alignment_depth",
    "rsi_gate",
    "above_ma5",
    "days_since_ma5_cross",
    "tier",
    "data_quality",
    "bars_available",
    "computed_at",
]

#: Rendering only. K1-K3 are pure alignment; K4-K6 add the RSI gate.
#: Verified against the two documented draft outputs — see the tests.
_CRITERIA = [
    ("K1", 5, False),
    ("K2", 4, False),
    ("K3", 3, False),
    ("K4", 5, True),
    ("K5", 4, True),
    ("K6", 3, True),
]


def render_criteria(alignment_depth: int, rsi_gate: Optional[bool]) -> list[str]:
    """Render the six legacy labels from the two stored axes.

    Kept as a display helper so the UI can still speak in "Kriteria" terms
    without those labels ever being the source of truth.
    """
    fired = []
    for label, needed_depth, needs_gate in _CRITERIA:
        if alignment_depth >= needed_depth and (not needs_gate or rsi_gate is True):
            fired.append(label)
    return fired


def alignment_depth(row_or_frame: pd.DataFrame) -> pd.Series:
    """How deep the MA stack is correctly ordered, given what exists.

    A comparison against a NaN MA is False, so a stock that simply lacks the
    history can never be promoted to a deeper tier by accident.
    """
    f = row_or_frame
    depth3 = (f["ma5"] > f["ma20"]) & (f["ma20"] > f["ma50"])
    depth4 = depth3 & (f["ma50"] > f["ma100"])
    depth5 = depth4 & (f["ma100"] > f["ma200"])

    depth = pd.Series(0, index=f.index, dtype="int64")
    depth[depth3.fillna(False)] = 3
    depth[depth4.fillna(False)] = 4
    depth[depth5.fillna(False)] = 5
    return depth


def days_since_ma5_cross(close: pd.Series, ma5: pd.Series) -> pd.Series:
    """Bars since price last crossed up through MA5.

    Trailing only — each bar looks backwards at crossings that had already
    happened, never at the whole series. NULL when price is currently below
    MA5, or when no crossing has occurred yet in the stored history.
    """
    close = close.reset_index(drop=True)
    ma5 = ma5.reset_index(drop=True)

    above = (close > ma5).fillna(False)
    previously_above = above.shift(1, fill_value=False)
    crossed_up = above & ~previously_above

    positions = np.arange(len(close), dtype="float64")
    cross_positions = pd.Series(np.where(crossed_up, positions, np.nan)).ffill()

    days = pd.Series(positions - cross_positions)
    days[~above] = np.nan
    return days.astype("Float64").astype("Int64")


def _tier(depth: int, gate: Optional[bool]) -> Optional[str]:
    """A grade derived from the two axes. Carries no probability.

    Nothing here implies a success rate — that would be an unmeasured
    percentage, and those are not allowed to reach the user until the
    calibration report exists.
    """
    if depth == 0:
        return None
    if gate is True:
        return {5: "A", 4: "B", 3: "C"}[depth]
    return {5: "B", 4: "C", 3: "D"}[depth]


def compute_for_ticker(
    indicators: pd.DataFrame,
    *,
    quality: Optional[pd.DataFrame] = None,
    computed_at: Optional[dt.datetime] = None,
) -> pd.DataFrame:
    """Signals for one ticker, for EVERY historical date.

    Not just today — the backtest replays this table, so a signal that only
    exists as a "current view" would leave nothing to measure.
    """
    if indicators is None or indicators.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    data = indicators.sort_values("date").reset_index(drop=True)

    out = pd.DataFrame({"ticker": data["ticker"], "date": data["date"]})
    out["alignment_depth"] = alignment_depth(data)

    rsi = pd.to_numeric(data["rsi14"], errors="coerce")
    # NULL where RSI is undefined — an unknown gate is not a failed gate.
    out["rsi_gate"] = pd.Series(
        np.where(rsi.isna(), None, rsi > RSI_GATE_LEVEL), index=data.index
    ).astype("object")

    close = pd.to_numeric(data["close"], errors="coerce")
    ma5 = pd.to_numeric(data["ma5"], errors="coerce")
    out["above_ma5"] = (close > ma5).where(ma5.notna(), None).astype("object")
    out["days_since_ma5_cross"] = days_since_ma5_cross(close, ma5)

    bars = pd.to_numeric(data["bars_available"], errors="coerce").fillna(0).astype("int64")
    out["bars_available"] = bars

    # Cap depth by what the history could actually support. Belt and braces:
    # the NaN comparisons above already prevent promotion, but an explicit cap
    # makes the rule impossible to lose to a future refactor.
    for depth, needed in sorted(DEPTH_REQUIREMENTS.items(), reverse=True):
        too_short = (out["alignment_depth"] >= depth) & (bars < needed)
        out.loc[too_short, "alignment_depth"] = depth - 1 if depth > 3 else 0
    out["alignment_depth"] = out["alignment_depth"].where(
        out["alignment_depth"].isin([0, 3, 4, 5]), 0
    )

    out["tier"] = [
        _tier(int(d), g) for d, g in zip(out["alignment_depth"], out["rsi_gate"])
    ]
    out["data_quality"] = _data_quality(
        bars, data["date"], quality, depth=out["alignment_depth"]
    )
    out["computed_at"] = computed_at or dt.datetime.now()

    return out.loc[:, SIGNAL_COLUMNS]


def bars_since_suspicious_jump(dates: pd.Series, quality: pd.DataFrame) -> pd.Series:
    """How many bars back the most recent contaminating bar was. NaN if none yet.

    Positional, and strictly backward-looking: bar 0 is the bar itself. An
    impossible bar (arithmetically corrupt OHLC) poisons the moving averages
    exactly as an unrecorded split does, so it counts here too when present.
    """
    q = quality.set_index("date")
    bad = q["suspicious_jump"].fillna(False).astype(bool)
    if "impossible_bar" in q.columns:
        bad = bad | q["impossible_bar"].fillna(False).astype(bool)
    jump = dates.map(bad).fillna(False).astype(bool).to_numpy()

    positions = np.arange(len(jump), dtype="float64")
    last_jump = pd.Series(np.where(jump, positions, np.nan)).ffill()
    return pd.Series(positions - last_jump.to_numpy(), index=dates.index)


def _data_quality(
    bars: pd.Series,
    dates: pd.Series,
    quality: Optional[pd.DataFrame],
    depth: Optional[pd.Series] = None,
) -> pd.Series:
    """Resolve the row's quality state.

    Precedence: insufficient_history > suspect > stale > ok.

    `insufficient_history` wins because it describes whether the alignment
    verdict itself could be reached, which matters more than the condition of a
    bar we were never able to fully judge.

    **Contamination.** A suspicious bar does not only spoil itself. A single
    unrecorded split or bad print sits inside every moving average that spans
    it, so the days *after* it are wrong too while reading `ok`. Observed on
    real data: ELTY.JK printed 5000 on isolated days while trading flat at 500,
    and for three days afterwards showed a full depth-5 alignment with MA5 at
    1400 and no quality flag at all.

    So the suspect mark propagates forward for as many bars as the deepest
    moving average that verdict relies on — 200 for depth 5, 100 for depth 4,
    50 for depth 3, and 5 otherwise (MA5 still feeds `above_ma5`).

    RSI is smoothed recursively and so is technically contaminated for longer,
    but its influence decays geometrically; the MA window is the honest,
    bounded rule.
    """
    deepest = max(DEPTH_REQUIREMENTS.values())
    state = pd.Series("ok", index=bars.index, dtype="object")

    if quality is not None and not quality.empty:
        q = quality.set_index("date")
        stale = dates.map(q["stale_price"]).fillna(False).astype(bool)
        state[stale.values] = "stale"

        since = bars_since_suspicious_jump(dates, quality)
        if depth is None:
            window = pd.Series(deepest, index=bars.index)
        else:
            window = depth.map(DEPTH_REQUIREMENTS).fillna(MA5_WINDOW)
        contaminated = since.notna() & (since <= window.to_numpy())
        state[contaminated.to_numpy()] = "suspect"

    state[(bars < deepest).values] = "insufficient_history"
    return state


def compute_all(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    if tickers is None:
        rows = con.execute("SELECT DISTINCT ticker FROM indicators ORDER BY ticker").fetchall()
        tickers = [r[0] for r in rows]

    has_quality = bool(
        con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'price_quality'"
        ).fetchone()[0]
    )

    frames = []
    for ticker in tickers:
        indicators = con.execute(
            "SELECT * FROM indicators WHERE ticker = ? ORDER BY date", [ticker]
        ).df()
        if indicators.empty:
            continue
        quality = None
        if has_quality:
            quality = con.execute(
                "SELECT date, stale_price, suspicious_jump, impossible_bar "
                "FROM price_quality WHERE ticker = ? ORDER BY date",
                [ticker],
            ).df()
        frames.append(compute_for_ticker(indicators, quality=quality))

    if not frames:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def upsert_signals(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> int:
    if frame is None or frame.empty:
        return 0
    payload = frame.loc[:, SIGNAL_COLUMNS].drop_duplicates(
        subset=["ticker", "date"], keep="last"
    )
    con.register("_incoming_signals", payload)
    try:
        con.execute(
            """
            INSERT INTO signals
                (ticker, date, alignment_depth, rsi_gate, above_ma5,
                 days_since_ma5_cross, tier, data_quality, bars_available, computed_at)
            SELECT ticker, date, alignment_depth, rsi_gate, above_ma5,
                   days_since_ma5_cross, tier, data_quality, bars_available, computed_at
            FROM _incoming_signals
            ON CONFLICT (ticker, date) DO UPDATE SET
                alignment_depth      = excluded.alignment_depth,
                rsi_gate             = excluded.rsi_gate,
                above_ma5            = excluded.above_ma5,
                days_since_ma5_cross = excluded.days_since_ma5_cross,
                tier                 = excluded.tier,
                data_quality         = excluded.data_quality,
                bars_available       = excluded.bars_available,
                computed_at          = excluded.computed_at
            """
        )
    finally:
        con.unregister("_incoming_signals")
    return len(payload)
