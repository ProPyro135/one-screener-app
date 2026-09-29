"""Hard TP & BEP Stop strategy (strategy I, EOD).

A port of the Pine Script "Market Structure: Hard TP & BEP Stop". Entries are
PINESCRIPT F's (``early_entry``); the exit is simpler:

    Stop        the base low (the lowest low of the 20 bars before the
                signal); once a bar's high reaches the buy price + 5%, the
                stop moves up to the buy price (BEP)
    TP          the whole position at the buy price + 10%

Both are resting orders from the bar after the fill, filled at their level or
at the open on a gap through; a bar that reaches both counts as the stop.

As the Pine is written it never exits: its ``else`` branch (no position yet)
wipes ``entryPrice`` and ``activeStop`` on the signal bar, and both are only
set on a signal bar, so the TP, the BEP trigger and the stop stay ``na`` and
no exit order is ever placed. Ported as its comments intend instead, with the
buy price taken as the actual fill (the next open) rather than the signal
close, so a TP is always over the price paid.

A stop exit over the buy price counts as TP, at or under it as CL (a BEP exit
at the buy price is a small loss after fees).

A *monitoring* screener with no validated edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idxcore.compute.early_entry import MIN_BARS, _all_bars, _ticker_meta, category_of, prepare  # noqa: F401

TP_PCT = 10.0
BEP_TRIGGER_PCT = 5.0


def run_state_machine(df: pd.DataFrame) -> tuple[list[str], list[dict], dict]:
    """Signals at the close, fills at the next open, TP and stop resting."""
    dates = df["date"].to_numpy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    signal = df["signal"].to_numpy(bool)
    range_low = df["range_low"].to_numpy(float)

    n = len(df)
    codes = [""] * n
    trades: list[dict] = []
    position_arr = [0] * n

    in_pos = pending_buy = orders = at_bep = False
    entry_price = stop = next_stop = np.nan
    entry_date = None
    entry_idx = -1

    def close_trade(i: int, code: str, price: float) -> None:
        codes[i] = code
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": "BUY Early", "exit_date": dates[i],
            "exit_price": price, "exit_code": code,
            "bars_held": i - entry_idx,
            "gross_return_pct": (price / entry_price - 1.0) * 100.0,
            "resolved": True,
        })

    for i in range(n):
        # 1. The BUY from the previous close fills at this open.
        if pending_buy:
            pending_buy, in_pos, orders, at_bep = False, True, False, False
            entry_price, entry_date, entry_idx = o[i], dates[i], i
            stop = next_stop
            codes[i] = "BUY Early"
        # 2. Resting exits, placed at an earlier close. The stop first.
        if in_pos and orders:
            if lo[i] <= stop:
                px = min(o[i], stop)
                in_pos = False
                if px > entry_price:
                    close_trade(i, "TP Stop", px)
                else:
                    close_trade(i, "CL BEP stop" if at_bep else "CL Stop loss", px)
            elif h[i] >= entry_price * (1 + TP_PCT / 100):
                in_pos = False
                close_trade(i, f"TP Target {TP_PCT:g}%", max(o[i], entry_price * (1 + TP_PCT / 100)))
        # 3. At the close: move the stop to BEP once +5% has printed.
        if in_pos:
            if h[i] >= entry_price * (1 + BEP_TRIGGER_PCT / 100) and not at_bep:
                stop, at_bep = max(stop, entry_price), True
            orders = True
        elif signal[i]:
            pending_buy, next_stop = True, range_low[i]
        position_arr[i] = int(in_pos)

    if in_pos:
        last = n - 1
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_code": "BUY Early", "exit_date": dates[last],
            "exit_price": c[last], "exit_code": "OPEN",
            "bars_held": last - entry_idx,
            "gross_return_pct": (c[last] / entry_price - 1.0) * 100.0,
            "resolved": False,
        })

    # A buy signalled at the last close, to be filled at tomorrow's open.
    setup_arr = [False] * n
    if n and pending_buy:
        setup_arr[-1] = True
    return codes, trades, {"position": position_arr, "setup": setup_arr}


if __name__ == "__main__":
    # Trade 1: signal at bar 1 (base low 95), fill at bar 2 (100). Bar 3
    # prints 106 -> stop to BEP 100; bar 4 dips to 99 -> CL at 100.
    # Trade 2: signal at bar 5, fill at bar 6 (100); bar 7 reaches 110 -> TP.
    # Trade 3: signal at bar 8, fill at bar 9; bar 10 gaps to 94 -> CL at 94.
    n = 12
    f = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": [100.0] * n, "high": [102.0] * n, "low": [98.5] * n, "close": [100.0] * n,
        "signal": [False] * n, "range_low": [95.0] * n,
    })
    f.loc[[1, 5, 8], "signal"] = True
    f.loc[3, "high"] = 106.0
    f.loc[4, "low"] = 99.0
    f.loc[7, "high"] = 111.0
    f.loc[10, ["open", "low"]] = [94.0, 93.0]
    codes, trades, lines = run_state_machine(f)
    got = [(t["exit_code"], round(t["exit_price"], 6)) for t in trades]
    assert got == [("CL BEP stop", 100.0), ("TP Target 10%", 110.0), ("CL Stop loss", 94.0)], got
    assert all(category_of(t["exit_code"]) in ("TAKE PROFIT", "CUT LOSS") for t in trades)
    assert not any(lines["setup"])
    print("early_entry_base_stop (BEP) self-check ok:", [(i, x) for i, x in enumerate(codes) if x])
