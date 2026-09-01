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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxcore.charts import PLOTLY_CONFIG, build_combined_figure  # noqa: E402
import trade_table  # noqa: E402
from idxcore.compute import bottom_fishing as bf  # noqa: E402
from idxcore.compute import trade_log as tl  # noqa: E402
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
# trade log — every Bottom Fishing BUY through to its TP/CL, newest first
# ---------------------------------------------------------------------------

# ~25s to walk every ticker through the state machine, against a store that
# changes once a trading day, so the cache is held for an hour rather than 5min.
@st.cache_data(ttl=3600, show_spinner="Menyusun tabel trade…")
def _log(version: str):
    with _connection() as con:
        return None if con is None else tl.build(con, bf)


st.subheader(t("sr_screener", lang))
try:
    log = _log(db._read_marker(db.SLIM_VERSION_MARKER))
except StoreBusy:
    log = None
    st.info(t("store_busy", lang), icon="⏳")

if log is not None and not log.empty:
    trade_table.render(log, lang, key="bf")

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
