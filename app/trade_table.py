"""The trade table both screener pages render, and its filters.

One module rather than a copy on each page: the two screeners differ only in
which state machine produced the log, and a table this wide is exactly the kind
of thing that drifts when it exists twice.

The table answers "when did this fire and what has it done since", so it is
sorted newest BUY first and every filter narrows toward a fresh entry.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from idxcore.compute import trade_log as tl
from idxcore.i18n import t

#: Status colours: blue = running, orange = armed, green = took profit,
#: red = stopped out. Same blue/orange the radar used for naik/pantau.
STATUS_COLOUR = {
    tl.OPEN: "#2962FF", tl.WATCHLIST: "#FB8C00",
    tl.CLOSED: "#2E7D32", tl.EXIT: "#C62828",
}
STATUS_ORDER = [tl.OPEN, tl.WATCHLIST, tl.CLOSED, tl.EXIT]


def _filters(cur: pd.DataFrame, lang: str, key: str) -> pd.DataFrame:
    """The four filters, applied to the one-row-per-ticker view."""
    dates = pd.to_datetime(cur["buy_date"]).dropna()
    c1, c2 = st.columns([2, 3])
    with c1:
        # A range picker rather than two boxes: the question is always "bought
        # between X and Y", never one open-ended bound.
        span = st.date_input(
            t("tl_f_dates", lang),
            value=(dates.min().date(), dates.max().date()) if len(dates) else (),
            min_value=dates.min().date() if len(dates) else None,
            max_value=dates.max().date() if len(dates) else None,
            key=f"{key}_dates",
        )
    with c2:
        picked_status = st.multiselect(
            t("tl_f_status", lang),
            [s for s in STATUS_ORDER if s in set(cur["status"])],
            key=f"{key}_status",
        )
    codes = [c for c in sorted(set(cur["entry_code"].dropna())) if c]
    picked_code = st.multiselect(t("tl_f_signal", lang), codes, key=f"{key}_code")
    skip_sleepy = st.checkbox(t("tl_f_sleepy", lang), value=True, key=f"{key}_sleepy")

    r = cur
    # A half-picked range is one date; leave the rows alone until both are set.
    if isinstance(span, (tuple, list)) and len(span) == 2:
        lo, hi = pd.Timestamp(span[0]), pd.Timestamp(span[1])
        bd = pd.to_datetime(r["buy_date"])
        # Watchlist rows have no BUY date at all. Dropping them here would mean
        # the date filter silently hides every armed setup, which is the
        # opposite of useful — exclude them with the status filter instead.
        r = r[bd.between(lo, hi) | bd.isna()]
    if picked_status:
        r = r[r["status"].isin(picked_status)]
    if picked_code:
        r = r[r["entry_code"].isin(picked_code)]
    if skip_sleepy:
        r = r[~(r["turnover"] < tl.SLEEPY_TURNOVER)]
    return r


def _style(frame: pd.DataFrame, lang: str):
    """Rename to the owner's column names, colour status and the two % columns."""
    out = pd.DataFrame({
        t("c_ticker", lang): frame["idx_code"].fillna(frame["ticker"]),
        t("c_name", lang): frame["name"],
        t("tl_status", lang): frame["status"],
        t("tl_buy_date", lang): pd.to_datetime(frame["buy_date"]),
        t("tl_pb", lang): frame["pb"],
        t("tl_buy_price", lang): frame["buy_price"],
        t("tl_last", lang): frame["last_close"],
        t("tl_fl", lang): frame["fl_pct"],
        t("tl_hi", lang): frame["hi_price"],
        t("tl_max_fl", lang): frame["max_fl_pct"],
        t("tl_exit_date", lang): pd.to_datetime(frame["exit_date"]),
        t("tl_exit_price", lang): frame["exit_price"],
        t("tl_pl", lang): frame["pl_pct"],
        t("tl_entry_code", lang): frame["entry_code"],
        t("tl_exit_code", lang): frame["exit_code"],
    })
    pct_cols = [t("tl_fl", lang), t("tl_max_fl", lang), t("tl_pl", lang)]
    price_cols = [t("tl_buy_price", lang), t("tl_last", lang),
                  t("tl_hi", lang), t("tl_exit_price", lang)]

    def paint_status(v):
        colour = STATUS_COLOUR.get(v)
        return f"background-color: {colour}; color: white; font-weight: 600" if colour else ""

    def paint_pct(v):
        # OPEN vs BULLISH/BEARISH: the status stays one word and the sign of the
        # floating return carries the direction instead.
        if pd.isna(v):
            return ""
        return "color: #2E7D32; font-weight: 600" if v >= 0 else "color: #C62828; font-weight: 600"

    # One format() call, not four. Styler.format() with no `subset` applies to
    # every column, so a second call resets the first call's formatters back to
    # the default — chaining them left prices as 92.000000 and dates as
    # 2026-08-31 00:00:00 on the live site while the colours worked fine.
    fmt = {c: "{:,.0f}" for c in price_cols}
    fmt.update({c: "{:+.2f}%" for c in pct_cols})
    fmt[t("tl_buy_date", lang)] = "{:%d/%m/%Y}"
    fmt[t("tl_exit_date", lang)] = "{:%d/%m/%Y}"
    return (out.style
            .map(paint_status, subset=[t("tl_status", lang)])
            .map(paint_pct, subset=pct_cols)
            .format(fmt, na_rep="—"))


def render(log: pd.DataFrame, lang: str, *, key: str) -> None:
    """Filters, the one-row-per-ticker table, and one ticker's history on click."""
    cur = tl.latest(log)
    r = _filters(cur, lang, key)
    # Newest entry first — the whole point of adding the date. Watchlist rows
    # have no BUY date and sort to the bottom.
    r = r.sort_values("buy_date", ascending=False, na_position="last")

    st.write(t("tl_count", lang, n=len(r)))
    if r.empty:
        st.info(t("tl_empty", lang))
        return
    st.caption(t("tl_pick", lang))

    event = st.dataframe(
        _style(r, lang), use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key=f"{key}_table",
    )
    st.caption(t("tl_legend", lang))
    st.caption(t("tl_gross", lang))

    picked = event.selection.rows if event and event.selection else []
    if picked:
        row = r.iloc[picked[0]]
        hist = log[log["ticker"] == row["ticker"]].sort_values(
            "buy_date", ascending=False, na_position="first")
        st.subheader(t("tl_history", lang, name=row["idx_code"] or row["ticker"]))
        st.dataframe(_style(hist, lang), use_container_width=True, hide_index=True)
