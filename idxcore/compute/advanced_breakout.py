"""Advanced Breakout strategy (EOD): the accumulation breakout, stricter.

A port of the Pine Script "Market Structure: Advanced Breakout" at its
defaults. It is ``accumulation_breakout`` with other settings, so it runs on
that module's state machine:

    Base           the prior 20 bars span <= 12% of their low, close > EMA200
    BUY Breakout   a close crossing over the prior 20-bar high, the bar after
                   a base bar, on volume >= 1.5x its 50-bar average, closing
                   in the top 30% of its bar
    Trailing stop  starts at max(close - 4 ATR, the base low), then the
                   highest high since the BUY minus 4 ATR, only ever rising;
                   a close under it sells at the next open

Fills, the TP/CL split and the WATCHLIST row are as in
``accumulation_breakout``. A *monitoring* screener with no validated edge.
"""

from __future__ import annotations

import pandas as pd

from idxcore.compute import accumulation_breakout as ab
from idxcore.compute.accumulation_breakout import _all_bars, _ticker_meta, category_of  # noqa: F401

E = ab.Params(accum_len=20, max_box_pct=12.0, vol_len=50, vol_mult=1.5, ema_len=200,
              ema_floor=1.0, ema_strict=True, atr_mult=4.0, strong_candle=True,
              stop_at_base_low=True)

MIN_BARS = E.ema_len + 1


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    return ab.prepare(history, E)


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    return ab.run_state_machine(df, E)
