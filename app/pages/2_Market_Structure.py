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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trade_table  # noqa: E402
from idxcore.compute import market_structure as ms  # noqa: E402
from idxcore.compute import trade_log as tl  # noqa: E402
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


# The build walks every ticker's bars through the state machine — ~22s — and the
# store only changes once a trading day, so the short radar TTL was wasteful.
@st.cache_data(ttl=3600, show_spinner="Menyusun tabel trade…")
def _log():
    with _connection() as con:
        return None if con is None else tl.build(con, ms)


st.subheader(t("ms_screener", lang))
try:
    log = _log()
except StoreBusy:
    st.info(t("store_busy", lang), icon="⏳")
    st.stop()

if log is None or log.empty:
    st.error(t("no_store", lang, path=DB_PATH))
    st.stop()

trade_table.render(log, lang, key="ms")
