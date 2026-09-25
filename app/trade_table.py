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
#: Quick periods, counted back from the newest BUY date. None = no lower bound
#: ("all") or the calendar picker ("custom").
PERIODS = {
    "1w": pd.DateOffset(weeks=1), "1m": pd.DateOffset(months=1),
    "3m": pd.DateOffset(months=3), "6m": pd.DateOffset(months=6),
    "1y": pd.DateOffset(years=1), "3y": pd.DateOffset(years=3),
    "5y": pd.DateOffset(years=5),
    "all": None, "custom": None,
}
#: Labels kept here, not in idxcore/i18n.py: a Streamlit hot reload re-imports
#: this file but keeps the old i18n module, so new i18n keys raise KeyError
#: until someone reboots the app.
PERIOD_TITLE = "PERIOD"
PERIOD_LABELS = {
    "en": {"1w": "1W", "1m": "1M", "3m": "3M", "6m": "6M", "1y": "1Y", "3y": "3Y",
           "5y": "5Y", "all": "All", "custom": "Custom"},
    "id": {"1w": "1 Mgg", "1m": "1 Bln", "3m": "3 Bln", "6m": "6 Bln", "1y": "1 Thn",
           "3y": "3 Thn", "5y": "5 Thn", "all": "Semua", "custom": "Custom"},
}
#: The backtest view lists at most this many trades, newest first. Styling tens
#: of thousands of rows would lag the page; the summary still counts them all.
MAX_ROWS = 500
TEXT = {
    "en": {"view": "VIEW", "now": "Current positions", "bt": "Backtest",
           "trades": "Trades", "tp": "TP", "cl": "CL", "open": "Still open",
           "win": "Win rate", "avg": "Average P&L",
           "avg_tp": "Average TP", "avg_cl": "Average CL",
           "none": "No closed trade in this selection.",
           "note": ("Measured on every trade in the selection, bought from {first} to "
                    "{last}. Gross: broker fees are not deducted. Win rate and P&L count "
                    "closed trades only (TP + CL). The sleepy-stock filter uses today's "
                    "turnover, not the turnover at the time of the trade."),
           "capped": "Showing the newest {shown:,} of {total:,} trades; the summary above counts all of them."},
    "id": {"view": "TAMPILAN", "now": "Posisi terkini", "bt": "Backtest",
           "trades": "Trade", "tp": "TP", "cl": "CL", "open": "Masih open",
           "win": "Win rate", "avg": "Rata-rata P&L",
           "avg_tp": "Rata-rata TP", "avg_cl": "Rata-rata CL",
           "none": "Tidak ada trade yang sudah selesai di pilihan ini.",
           "note": ("Diukur dari semua trade di pilihan ini, BUY dari {first} sampai "
                    "{last}. Bruto: fee broker belum dipotong. Win rate dan P&L hanya "
                    "menghitung trade yang sudah selesai (TP + CL). Filter saham tidur "
                    "memakai nilai transaksi hari ini, bukan saat trade terjadi."),
           "capped": "Menampilkan {shown:,} trade terbaru dari {total:,}; ringkasan di atas menghitung semuanya."},
}


def _filters(cur: pd.DataFrame, lang: str, key: str, default: str = "all") -> pd.DataFrame:
    """PERIOD, status and sleepy filters over whichever rows are passed in."""
    dates = pd.to_datetime(cur["buy_date"]).dropna()
    # One click for the usual windows; the range calendar only for Custom. Its
    # own row, so all the buttons fit on one line.
    period = st.segmented_control(
        PERIOD_TITLE, list(PERIODS), default=default,
        format_func=lambda p: PERIOD_LABELS.get(lang, PERIOD_LABELS["en"])[p],
        key=f"{key}_period",
    )
    span = ()
    if len(dates) and period == "custom":
        with st.columns(2)[0]:
            span = st.date_input(
                PERIOD_TITLE, label_visibility="collapsed",
                value=(dates.min().date(), dates.max().date()),
                min_value=dates.min().date(), max_value=dates.max().date(),
                key=f"{key}_dates",
            )
    elif len(dates) and PERIODS.get(period) is not None:
        hi = dates.max()
        span = ((hi - PERIODS[period]).date(), hi.date())
    with st.columns(2)[0]:
        picked_status = st.multiselect(
            t("tl_f_status", lang),
            [s for s in STATUS_ORDER if s in set(cur["status"])],
            key=f"{key}_status",
        )
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


def _backtest(log: pd.DataFrame, lang: str, key: str) -> None:
    """Every trade bought in the period, a measured summary, and the newest rows."""
    tx = TEXT.get(lang, TEXT["en"])
    # Delisted names stay in: a backtest that drops the failures flatters itself.
    r = _filters(log[log["status"] != tl.WATCHLIST], lang, f"{key}_bt", default="1y")
    r = r.sort_values("buy_date", ascending=False)
    done = r[r["status"].isin([tl.CLOSED, tl.EXIT])]
    pl = done["pl_pct"]

    def avg(s: pd.Series) -> str:
        return f"{s.mean():+.2f}%" if len(s) else "—"

    # Average P&L is the mean over every CLOSED and EXIT trade. The TP and CL
    # averages beside it show how a low win rate can still average a profit.
    tp_pl = done.loc[done["status"] == tl.CLOSED, "pl_pct"]
    cl_pl = done.loc[done["status"] == tl.EXIT, "pl_pct"]
    c = st.columns(4)
    c[0].metric(tx["trades"], f"{len(r):,}")
    c[1].metric(tx["tp"], f"{len(tp_pl):,}")
    c[2].metric(tx["cl"], f"{len(cl_pl):,}")
    c[3].metric(tx["win"], f"{(pl > 0).mean() * 100:.1f}%" if len(pl) else "—")
    c = st.columns(4)
    c[0].metric(tx["avg"], avg(pl))
    c[1].metric(tx["avg_tp"], avg(tp_pl))
    c[2].metric(tx["avg_cl"], avg(cl_pl))
    if r.empty:
        st.info(tx["none"])
        return
    bd = pd.to_datetime(r["buy_date"])
    st.caption(f"{tx['open']}: {(r['status'] == tl.OPEN).sum():,} · "
               + tx["note"].format(first=f"{bd.min():%d/%m/%Y}", last=f"{bd.max():%d/%m/%Y}"))
    if done.empty:
        st.info(tx["none"])

    st.dataframe(_style(r.head(MAX_ROWS), lang), use_container_width=True,
                 hide_index=True, placeholder="—")
    if len(r) > MAX_ROWS:
        st.caption(tx["capped"].format(shown=MAX_ROWS, total=len(r)))
    st.caption(t("tl_legend", lang))


def render(log: pd.DataFrame, lang: str, *, key: str) -> None:
    """Current positions (one row per ticker, history on click) or the backtest."""
    tx = TEXT.get(lang, TEXT["en"])
    mode = st.segmented_control(
        tx["view"], ["now", "bt"], default="now",
        format_func=lambda m: tx[m], key=f"{key}_mode",
    ) or "now"
    if mode == "bt":
        _backtest(log, lang, key)
        return

    cur = tl.latest(log)
    if "is_active" in cur.columns:
        # Where each stock stands today only makes sense for listed stocks.
        cur = cur[cur["is_active"]]
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
        _style(r, lang), use_container_width=True, hide_index=True, placeholder="—",
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
        st.dataframe(_style(hist, lang), use_container_width=True, hide_index=True,
                     placeholder="—")
