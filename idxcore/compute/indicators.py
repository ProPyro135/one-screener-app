"""Indicator library (IVI-74).

MA5/20/50/100/200, RSI(14) and MACD(12,26,9). **Nothing else in the system may
compute these.** If the dashboard needs RSI it reads `indicators.rsi14`.

The bug being fixed
-------------------
The draft used a simple rolling mean::

    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()

That is Cutler's RSI. The standard — TradingView, and ``TTR::RSI`` in the
original Shiny app — is Wilder's smoothing. Criteria 4, 5 and 6 are gated
entirely on ``RSI > 50``, so the two formulas disagree about whether a stock
qualifies on a meaningful share of days.

On verification
---------------
The issue asks for a golden-file test against ``TTR::RSI`` to 1e-6, as the
contract between this module and the R app. There is no R app and no R runtime
on this machine, so that reference cannot be produced here. Verifying Wilder's
RSI only against this module's own output would prove nothing, so instead:

* ``_wilder_average_recursive`` implements the definition literally, one bar at
  a time, and the test asserts the vectorised path matches it exactly;
* the test suite also checks a hand-computed seed and first smoothed value, and
  the documented edge cases (all-gains, all-losses, flat).

If R ever comes back, add the TTR golden file — the vectorised implementation
is already seeded the way TTR seeds it, so it should match.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

RSI_PERIOD = 14
MA_WINDOWS = (5, 20, 50, 100, 200)
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

INDICATOR_COLUMNS = [
    "ticker",
    "date",
    "close",
    "ma5",
    "ma20",
    "ma50",
    "ma100",
    "ma200",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "tenkan",
    "kijun",
    "senkou_a",
    "senkou_b",
    "donchian_upper",
    "donchian_lower",
    "bb_basis",
    "bb_upper",
    "bb_lower",
    "fcb_upper",
    "fcb_lower",
    "fcb_dot_upper",
    "fcb_dot_lower",
    "bars_available",
    "computed_at",
]


# ---------------------------------------------------------------------------
# Wilder's smoothing
# ---------------------------------------------------------------------------

def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed average: SMA seed, then recursive smoothing.

    Seeded the way TTR and TradingView seed it — the first value is the simple
    mean of the first `period` observations, and only then does the recursion
    start. A bare ``ewm(alpha=1/period)`` skips that seed and produces slightly
    different numbers through the warm-up.

    Everything before the seed is NaN, never 0. A warm-up row is an undefined
    indicator, and writing 0 there would read downstream as a real value.
    """
    v = pd.to_numeric(values, errors="coerce").astype("float64").reset_index(drop=True)
    valid = v.dropna()
    out = pd.Series(np.nan, index=v.index, dtype="float64")

    if len(valid) < period:
        return out

    seed_pos = valid.index[period - 1]
    seed = valid.iloc[:period].mean()

    # Feed ewm a series that begins at the seed. With adjust=False and
    # alpha = 1/period, ewm computes y[i] = ((period-1)*y[i-1] + x[i]) / period,
    # which is exactly Wilder's recursion.
    tail = v.loc[seed_pos:].copy()
    tail.iloc[0] = seed
    smoothed = tail.ewm(alpha=1.0 / period, adjust=False).mean()
    out.loc[seed_pos:] = smoothed
    return out


def _wilder_average_recursive(values: pd.Series, period: int) -> pd.Series:
    """The definition, written literally. Used by the tests to check the fast path."""
    v = pd.to_numeric(values, errors="coerce").astype("float64").reset_index(drop=True)
    out = pd.Series(np.nan, index=v.index, dtype="float64")

    seen: list[float] = []
    average: Optional[float] = None
    for i, x in enumerate(v):
        if np.isnan(x):
            continue
        if average is None:
            seen.append(x)
            if len(seen) == period:
                average = float(np.mean(seen))
                out.iloc[i] = average
        else:
            average = (average * (period - 1) + x) / period
            out.iloc[i] = average
    return out


def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI(14) with Wilder's smoothing.

    Edge cases, all of which the draft got wrong or warned on:

    * ``avg_loss == 0`` with gains present -> 100, not ``inf`` and not a warning.
    * ``avg_gain == 0`` with losses present -> 0.
    * both zero (a perfectly flat stretch) -> 50. There is no momentum either
      way, so neutral is the honest answer. Such rows are also flagged
      ``stale_price`` by the quality gates.
    * warm-up rows -> NaN, never 0.
    """
    series = pd.to_numeric(close, errors="coerce").astype("float64").reset_index(drop=True)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _wilder_average(gain, period)
    avg_loss = _wilder_average(loss, period)

    rsi = pd.Series(np.nan, index=series.index, dtype="float64")
    defined = avg_gain.notna() & avg_loss.notna()

    both_zero = defined & (avg_gain == 0) & (avg_loss == 0)
    no_loss = defined & (avg_loss == 0) & (avg_gain > 0)
    normal = defined & (avg_loss > 0)

    rs = avg_gain[normal] / avg_loss[normal]
    rsi[normal] = 100.0 - (100.0 / (1.0 + rs))
    rsi[no_loss] = 100.0
    rsi[both_zero] = 50.0

    rsi.index = close.index
    return rsi


# ---------------------------------------------------------------------------
# the rest
# ---------------------------------------------------------------------------

def moving_averages(close: pd.Series, windows=MA_WINDOWS) -> dict[str, pd.Series]:
    """Simple moving averages. NULL until the window is full, never 0."""
    series = pd.to_numeric(close, errors="coerce")
    return {
        f"ma{w}": series.rolling(window=w, min_periods=w).mean() for w in windows
    }


def macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    series = pd.to_numeric(close, errors="coerce")
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    *,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> dict[str, pd.Series]:
    """Tenkan, Kijun and the two Senkou spans, all UNSHIFTED.

    Ichimoku draws the spans 26 bars into the future. That shift belongs to the
    drawing, not the data: storing a value against a date it was not computed
    on would put future-dated rows in the table and read as lookahead to anyone
    auditing it later. The chart applies the shift.

    Every component is a midpoint of a trailing high/low window, so nothing
    here reads a bar later than the one it is stored against.
    """
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")

    def midpoint(period: int) -> pd.Series:
        top = high.rolling(period, min_periods=period).max()
        bottom = low.rolling(period, min_periods=period).min()
        return (top + bottom) / 2.0

    tenkan = midpoint(tenkan_period)
    kijun = midpoint(kijun_period)
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": (tenkan + kijun) / 2.0,
        "senkou_b": midpoint(senkou_b_period),
    }


def donchian(
    high: pd.Series, low: pd.Series, *, period: int = 20
) -> dict[str, pd.Series]:
    """Rolling N-bar high and low — the stepped channel.

    Trailing windows only. A full-series max would be a classic lookahead bug
    (invariant 7), and it is exactly the shape this indicator invites.
    """
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    return {
        "donchian_upper": high.rolling(period, min_periods=period).max(),
        "donchian_lower": low.rolling(period, min_periods=period).min(),
    }


#: Defaults from the Pine source this reproduces. Named rather than inlined so
#: a reader can check them against the script.
FCB_PATTERN = 1
FCB_MIN_DOT_HOLD = 5
BB_PERIOD = 20
BB_MULT = 2.0


def bollinger(
    close: pd.Series, *, period: int = BB_PERIOD, mult: float = BB_MULT
) -> dict[str, pd.Series]:
    """Bollinger Band, matching Pine's ``ta.stdev``.

    Pine's standard deviation is the population one. pandas defaults to the
    sample version, ``ddof=1``, which is a different number on every bar — the
    band would be visibly wider and no test comparing against TradingView would
    ever line up. ``ddof=0`` is the whole point of this function.
    """
    close = pd.to_numeric(close, errors="coerce")
    basis = close.rolling(period, min_periods=period).mean()
    dev = close.rolling(period, min_periods=period).std(ddof=0) * mult
    return {"bb_basis": basis, "bb_upper": basis + dev, "bb_lower": basis - dev}


def _raw_fractals(
    high: np.ndarray, low: np.ndarray, pattern: int
) -> tuple[np.ndarray, np.ndarray]:
    """The unmodified fractal levels, each carried forward until replaced.

    A fractal at ``pattern=1`` is the middle of three bars whose high beats both
    neighbours. It cannot be recognised until the bar after it, which is why the
    script draws the result two bars back — see the offset note in `charts.py`.
    """
    n = len(high)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    last_up = np.nan
    last_dn = np.nan
    span = pattern * 2 + 1

    for t in range(n):
        if t >= span:
            peak_up = high[t - (pattern + 1)]
            peak_dn = low[t - (pattern + 1)]
            ok_up = ok_dn = True
            for i in range(pattern, 0, -1):
                ok_up = ok_up and high[t - i] < high[t - i - 1]
                ok_dn = ok_dn and low[t - i] > low[t - i - 1]
            for i in range(pattern + 2, pattern * 2 + 2):
                ok_up = ok_up and high[t - i] < high[t - i + 1]
                ok_dn = ok_dn and low[t - i] > low[t - i + 1]
            if ok_up:
                last_up = peak_up
            if ok_dn:
                last_dn = peak_dn
        up[t] = last_up
        dn[t] = last_dn
    return up, dn


def _same(a: float, b: float) -> bool:
    """Pine treats two `na`s as not-different. NaN != NaN in Python does not."""
    if np.isnan(a) and np.isnan(b):
        return True
    return a == b


def fractal_chaos_bands(
    high: pd.Series,
    low: pd.Series,
    *,
    pattern: int = FCB_PATTERN,
    min_dot_hold: int = FCB_MIN_DOT_HOLD,
) -> dict[str, pd.Series]:
    """Classic Fractal Chaos Bands, plus the held levels.

    The upper band is the last confirmed up-fractal (a swing high) carried
    forward until a new one forms; the lower band is the last down-fractal. Each
    band is a *level*: it holds flat, then jumps when a new fractal lands, which
    is what makes the pair read as a clean stepped envelope with the upper above
    price and the lower below it — matching the owner's TradingView charts.

    An earlier port also applied the script's optional "bearish blue step-down",
    which drags a band toward price on a trend flip. Measured against the
    reference charts (BUMI/RATU/BNBR) that made the bands hug the candles and
    cross constantly, unlike the screenshots, so it is not applied here. The
    plain bands are what the charts show.

    **The dots.** When a band jumps to a new fractal, its previous value is
    frozen as a held level — but only if it had lasted at least
    ``min_dot_hold`` bars. That filter is why a handful of levels persist rather
    than one dot per fractal.

    Every value is a function of bars at or before its own date. The two-bar
    display shift the script applies is a drawing decision and lives in
    `charts.py`, so nothing stored here is future-dated.
    """
    h = pd.to_numeric(high, errors="coerce").to_numpy(dtype=float)
    l = pd.to_numeric(low, errors="coerce").to_numpy(dtype=float)
    n = len(h)

    up_raw, dn_raw = _raw_fractals(h, l, pattern)

    # The band IS the carried-forward fractal — no trend, no drag.
    upper = up_raw.astype(float).copy()
    lower = dn_raw.astype(float).copy()
    dot_up = np.full(n, np.nan)
    dot_dn = np.full(n, np.nan)

    prev_dot_up = np.nan
    prev_dot_dn = np.nan
    up_start = 0
    dn_start = 0

    for t in range(n):
        if t > 0 and not _same(up_raw[t], up_raw[t - 1]):
            if t - up_start >= min_dot_hold and not np.isnan(up_raw[t - 1]):
                prev_dot_up = up_raw[t - 1]
            up_start = t
        if t > 0 and not _same(dn_raw[t], dn_raw[t - 1]):
            if t - dn_start >= min_dot_hold and not np.isnan(dn_raw[t - 1]):
                prev_dot_dn = dn_raw[t - 1]
            dn_start = t
        dot_up[t] = prev_dot_up
        dot_dn[t] = prev_dot_dn

    index = high.index if hasattr(high, "index") else None
    return {
        "fcb_upper": pd.Series(upper, index=index),
        "fcb_lower": pd.Series(lower, index=index),
        "fcb_dot_upper": pd.Series(dot_up, index=index),
        "fcb_dot_lower": pd.Series(dot_dn, index=index),
    }


def compute_for_ticker(
    frame: pd.DataFrame,
    *,
    computed_at: Optional[dt.datetime] = None,
) -> pd.DataFrame:
    """Compute every indicator for one ticker's bars, ascending by date."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=INDICATOR_COLUMNS)

    data = frame.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(data["close"], errors="coerce")

    out = pd.DataFrame({"ticker": data["ticker"], "date": data["date"], "close": close})
    for name, series in moving_averages(close).items():
        out[name] = series

    out["rsi14"] = wilder_rsi(close)
    line, sig, hist = macd(close)
    out["macd"], out["macd_signal"], out["macd_hist"] = line, sig, hist

    # High/low based indicators. Missing columns leave them NULL rather than
    # silently substituting close, which would produce a plausible-looking
    # channel that is not the one the exchange traded.
    for name, series in bollinger(close).items():
        out[name] = series

    if "high" in data.columns and "low" in data.columns:
        for name, series in ichimoku(data["high"], data["low"]).items():
            out[name] = series
        for name, series in donchian(data["high"], data["low"]).items():
            out[name] = series
        for name, series in fractal_chaos_bands(
            data["high"], data["low"]
        ).items():
            out[name] = series
    else:
        for name in ("tenkan", "kijun", "senkou_a", "senkou_b",
                     "donchian_upper", "donchian_lower",
                     "fcb_upper", "fcb_lower", "fcb_dot_upper", "fcb_dot_lower"):
            out[name] = np.nan

    out["bars_available"] = np.arange(1, len(data) + 1)
    out["computed_at"] = computed_at or dt.datetime.now()
    return out.loc[:, INDICATOR_COLUMNS]


def compute_all(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute indicators for every ticker with stored prices.

    Always computes each ticker's full series rather than only the new tail.
    Wilder's smoothing is recursive, so today's RSI depends on every bar before
    it; recomputing only recent dates would produce a different number than a
    full rebuild and quietly break reproducibility. A full series is a few
    thousand rows, so correctness is cheap here.
    """
    if tickers is None:
        rows = con.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchall()
        tickers = [r[0] for r in rows]

    frames = []
    for ticker in tickers:
        prices = con.execute(
            "SELECT ticker, date, open, high, low, close FROM prices "
            "WHERE ticker = ? ORDER BY date",
            [ticker],
        ).df()
        if prices.empty:
            continue
        frames.append(compute_for_ticker(prices))

    if not frames:
        return pd.DataFrame(columns=INDICATOR_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def upsert_indicators(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> int:
    """Write indicators, keyed on (ticker, date).

    The column list is built from INDICATOR_COLUMNS rather than spelled out
    three times in the SQL. It used to be written by hand, which meant adding an
    indicator required editing an INSERT list, a SELECT list and an UPDATE list
    in step — and forgetting one of them stores nothing while reporting success,
    the quietest possible failure.
    """
    if frame is None or frame.empty:
        return 0
    payload = frame.loc[:, INDICATOR_COLUMNS].drop_duplicates(
        subset=["ticker", "date"], keep="last"
    )

    columns = list(INDICATOR_COLUMNS)
    keys = {"ticker", "date"}
    names = ", ".join(columns)
    separator = ",\n                "
    updates = separator.join(
        f"{c} = excluded.{c}" for c in columns if c not in keys
    )

    con.register("_incoming_indicators", payload)
    try:
        con.execute(
            f"""
            INSERT INTO indicators ({names})
            SELECT {names} FROM _incoming_indicators
            ON CONFLICT (ticker, date) DO UPDATE SET
                {updates}
            """
        )
    finally:
        con.unregister("_incoming_indicators")
    return len(payload)
