"""Market Structure — a second screener.

A radar over the whole active universe using the Market Structure auto-engine
(regime EMA20/SMA200 + swing pivots + dry-volume pullback breakout). Each stock
is coloured by its current state — RISING (in a position), WATCH (a pullback is
armed), WAIT (flat) — with its latest signal and reason. Read-only over the
store, so it runs unchanged on the full local store and the slim hosted one.
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

from idxcore.compute import market_structure as ms  # noqa: E402
from idxcore.i18n import LANGUAGES, default_language, t  # noqa: E402
from idxcore.store import db  # noqa: E402

st.set_page_config(page_title="Market Structure — IDX", page_icon="🏗️", layout="wide")


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


def _pick_language() -> str:
    if "lang" not in st.session_state:
        st.session_state["lang"] = default_language()
    codes = list(LANGUAGES)
    with st.sidebar:
        choice = st.radio(
            t("language", st.session_state["lang"]), options=codes,
            index=codes.index(st.session_state["lang"]),
            format_func=lambda c: LANGUAGES[c], horizontal=True, key="ms_lang",
        )
    st.session_state["lang"] = choice
    return choice


lang = _pick_language()

st.title(f"🏗️ {t('ms_title', lang)}")
st.caption(t("ms_caption", lang))


@st.cache_data(ttl=300, show_spinner="Menghitung radar…")
def _radar():
    with _connection() as con:
        return None if con is None else ms.radar_screen(con, lookback=5)


st.subheader(t("ms_screener", lang))
try:
    radar = _radar()
except StoreBusy:
    st.info(t("store_busy", lang), icon="⏳")
    st.stop()

if radar is None or radar.empty:
    st.error(t("no_store", lang, path=DB_PATH))
    st.stop()

st_label = {"naik": t("st_naik", lang), "pantau": t("st_pantau", lang),
            "tunggu": t("st_tunggu", lang)}
order = {"naik": 0, "pantau": 1, "tunggu": 2}
picked = st.multiselect(
    t("sr_status_filter", lang),
    [st_label[k] for k in ("naik", "pantau", "tunggu")],
    default=[st_label["naik"], st_label["pantau"]],
)
r = radar.copy()
r["_lbl"] = r["status"].map(st_label)
if picked:
    r = r[r["_lbl"].isin(picked)]
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
palette = {st_label[k]: ms.STATUS_COLOUR[k] for k in st_label}


def _paint(v):
    colour = palette.get(v)
    return f"background-color: {colour}; color: white; font-weight: 600" if colour else ""


styled = (tbl.style
          .map(_paint, subset=[t("sr_status", lang)])
          .format({t("c_close", lang): "{:,.0f}"}))
st.dataframe(styled, use_container_width=True, hide_index=True)
