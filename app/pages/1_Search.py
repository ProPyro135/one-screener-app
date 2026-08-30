"""Search — look up any saham by code or name.

A page on the One Screener app. Type a ticker or company name to pull up its
chart (the same combined figure the dashboard draws), its indicators, and the
criteria it currently meets. Read-only over the store, so it runs unchanged on
the full local store and the slim hosted one.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from idxcore.charts import PLOTLY_CONFIG, build_combined_figure  # noqa: E402
from idxcore.compute import bottom_fishing as bf  # noqa: E402
from idxcore.compute.signals import render_criteria  # noqa: E402
from idxcore.i18n import LANGUAGES, default_language, t  # noqa: E402
from idxcore.store import db, read  # noqa: E402

st.set_page_config(page_title="Search — IDX", page_icon="🔎", layout="wide")


def _resolve_db_path() -> str:
    env = os.environ.get("IDXCORE_DB")
    if env:
        return env
    full = Path(db.DEFAULT_DB_PATH)
    if full.exists():
        return str(full)
    # Hosted: no full store — fetch the slim snapshot from the Release asset.
    return str(db.ensure_slim_store())


DB_PATH = _resolve_db_path()

QUALITY_KEYS = {
    "ok": "q_ok", "insufficient_history": "q_short",
    "stale": "q_stale", "suspect": "q_suspect",
}


class StoreBusy(RuntimeError):
    """The nightly sync is writing; one writer or many readers, never both."""


@contextmanager
def _connection():
    if not Path(DB_PATH).exists():
        yield None
        return
    try:
        con = db.connect(DB_PATH, read_only=True)
    except (db.StoreLocked, duckdb.IOException) as exc:
        raise StoreBusy(str(exc)) from exc
    try:
        yield con
    finally:
        con.close()


@st.cache_data(ttl=60)
def _signals():
    with _connection() as con:
        return None if con is None else read.latest_signals(con)


def _pick_language() -> str:
    if "lang" not in st.session_state:
        st.session_state["lang"] = default_language()
    codes = list(LANGUAGES)
    with st.sidebar:
        choice = st.radio(
            t("language", st.session_state["lang"]), options=codes,
            index=codes.index(st.session_state["lang"]),
            format_func=lambda c: LANGUAGES[c], horizontal=True, key="sr_lang",
        )
    st.session_state["lang"] = choice
    return choice


def _pick_theme(lang: str) -> str:
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    labels = {"dark": t("theme_dark", lang), "light": t("theme_light", lang)}
    with st.sidebar:
        choice = st.radio(
            t("theme", lang), options=("dark", "light"),
            index=("dark", "light").index(st.session_state["theme"]),
            format_func=lambda c: labels[c], horizontal=True, key="sr_theme",
        )
    st.session_state["theme"] = choice
    return choice


lang = _pick_language()
theme = _pick_theme(lang)

st.title(f"🔎 {t('sr_title', lang)}")
st.caption(t("sr_caption", lang))

try:
    signals = _signals()
except StoreBusy:
    st.info(t("store_busy", lang), icon="⏳")
    st.stop()

if signals is None or signals.empty:
    st.error(t("no_store", lang, path=DB_PATH))
    st.stop()

# ---------------------------------------------------------------------------
# radar screener — each stock's current Bottom Fishing state, colour-coded
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Menghitung radar…")
def _radar():
    with _connection() as con:
        return None if con is None else bf.radar_screen(con)  # last signal ever


st.subheader(t("sr_screener", lang))
try:
    radar = _radar()
except StoreBusy:
    radar = None
    st.info(t("store_busy", lang), icon="⏳")

if radar is not None and not radar.empty:
    st_label = {"naik": t("st_naik", lang), "bottom": t("st_bottom", lang),
                "tunggu": t("st_tunggu", lang)}
    order = {"naik": 0, "bottom": 1, "tunggu": 2}
    picked = st.multiselect(
        t("sr_status_filter", lang),
        [st_label[k] for k in ("naik", "bottom", "tunggu")],
        default=[st_label["naik"], st_label["bottom"]],
    )
    # Second filter, on the signal itself: "show me every BUY (a2)" is a
    # different question from "show me everything armed". Options come from
    # REASON_MAP so the list cannot drift from the state machine's codes.
    no_signal = t("ms_no_signal", lang)
    present = set(radar["code"])
    sig_options = [c for c in bf.REASON_MAP if c in present]
    if (radar["code"] == "").any():
        sig_options.append(no_signal)
    picked_sig = st.multiselect(t("ms_signal_filter", lang), sig_options)

    r = radar.copy()
    r["_lbl"] = r["status"].map(st_label)
    if picked:
        r = r[r["_lbl"].isin(picked)]
    if picked_sig:
        mask = r["code"].isin(picked_sig)
        if no_signal in picked_sig:
            mask = mask | (r["code"] == "")
        r = r[mask]
    r = r.sort_values("status", key=lambda s: s.map(order))
    st.write(t("sr_screener_count", lang, n=len(r)))

    tbl = pd.DataFrame({
        t("c_ticker", lang): r["idx_code"].fillna(r["ticker"]),
        t("c_name", lang): r["name"],
        t("c_close", lang): r["close"],
        t("sr_status", lang): r["_lbl"],
        t("sr_signal", lang): r["code"],
        t("sr_reason", lang): r["reason"],
    })
    palette = {st_label[k]: bf.STATUS_COLOUR[k] for k in st_label}

    def _paint(v):
        colour = palette.get(v)
        return f"background-color: {colour}; color: white; font-weight: 600" if colour else ""

    styled = (tbl.style
              .map(_paint, subset=[t("sr_status", lang)])
              .format({t("c_close", lang): "{:,.0f}"}))
    st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()
st.subheader(f"🔎 {t('sr_lookup', lang)}")

# Native searchable dropdown: "BBCA — Bank Central Asia Tbk."
code = (signals["idx_code"] if "idx_code" in signals.columns else signals["ticker"])
code = code.fillna(signals["ticker"]).astype(str)
name = signals["name"].fillna(code).astype(str)
labels = dict(zip(code + " — " + name, signals["ticker"]))
choice = st.selectbox(t("sr_saham", lang), sorted(labels), index=None,
                      placeholder=t("sr_placeholder", lang))
if choice is None:
    st.stop()

ticker = labels[choice]
row = signals[signals["ticker"] == ticker].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric(t("c_close", lang), f"{row['close']:,.0f}" if pd.notna(row["close"]) else "—")
c2.metric(t("c_tier", lang), row["tier"] or "—")
c3.metric(t("c_depth", lang), int(row["alignment_depth"]))
c4.metric(t("c_rsi", lang), f"{row['rsi14']:.1f}" if pd.notna(row["rsi14"]) else "—")
st.caption(t("sr_meta", lang,
             crit=", ".join(render_criteria(int(row["alignment_depth"]), row["rsi_gate"])) or "—",
             board=row["board"] or "—", sector=row["sector"] or "—"))

if row["data_quality"] == "insufficient_history":
    st.warning(t("warn_short", lang, bars=int(row["bars_available"])), icon="⚠️")
elif row["data_quality"] == "suspect":
    st.error(t("warn_suspect", lang), icon="⚠️")
elif row["data_quality"] == "stale":
    st.warning(t("warn_stale", lang), icon="⚠️")

bars = st.slider(t("chart_bars", lang), 60, 1000, 250, step=10)
show_extras = st.checkbox(t("chart_show_extras", lang), value=False,
                          help=t("chart_extras_help", lang))

try:
    with _connection() as con:
        history = read.ticker_history(con, ticker, limit=bars) if con is not None else pd.DataFrame()
except StoreBusy:
    st.info(t("store_busy", lang), icon="⏳")
    st.stop()

if history.empty:
    st.info(t("chart_no_history", lang))
    st.stop()

st.plotly_chart(
    build_combined_figure(history, ticker, theme=theme, show_extras=show_extras),
    use_container_width=True, config=PLOTLY_CONFIG,
)
st.caption(t("chart_price_note", lang))
st.caption(t("chart_fitted_note", lang))
