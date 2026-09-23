"""Strategi — the owner's Pine Script strategies, one trade table.

Pick a strategy (PINESCRIPT A = Market Structure, B = Reversal Sniper) and the
same trade table shows every stock's latest trade under it: OPEN, WATCHLIST,
CLOSED or EXIT. Read-only over the store, so it runs unchanged on the full
local store and the slim hosted one.
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
from idxcore.compute import reversal_sniper as rs  # noqa: E402
from idxcore.compute import trade_log as tl  # noqa: E402
from idxcore.i18n import LANGUAGES, default_language, t  # noqa: E402
from idxcore.store import db  # noqa: E402

st.set_page_config(page_title="Strategi — IDX", page_icon="🏗️", layout="wide")

# Strings live here, not in idxcore/i18n.py: page code is re-run on every
# deploy, while a hot reload can keep an old i18n module in memory.
TITLE = {"en": "Pine Script strategies", "id": "Strategi Pine Script"}
STRATEGIES = {
    "A": (ms, "ms", {
        "en": ("Market Structure. Auto regime (EMA20 in a volatile swing, otherwise "
               "SMA200), a rebound off a higher swing low on dry volume, then a "
               "volume-backed breakout. TP only when the exit is above the buy price. "
               "Monitoring, not a proven signal."),
        "id": ("Market Structure. Regime otomatis (EMA20 saat swing volatil, selain "
               "itu SMA200), rebound dari swing-low yang lebih tinggi dengan volume "
               "kering, lalu breakout didukung volume. TP hanya kalau harga keluar di "
               "atas harga beli. Ini pemantauan, bukan sinyal terbukti."),
    }),
    "B": (rs, "rs", {
        "en": ("Reversal Sniper. After a 50-day low, a higher high and higher low, "
               "then a close above the high (BUY HH-HL); after a TP, a dip under MA20 "
               "and a close back above it (BUY Re-Entry). TP on the first close under "
               "MA5 after a rally, only when above the buy price. Monitoring, not a "
               "proven signal."),
        "id": ("Reversal Sniper. Setelah dasar 50 hari, terbentuk higher high dan "
               "higher low, lalu close di atas puncak (BUY HH-HL); setelah TP, harga "
               "turun ke bawah MA20 lalu close kembali di atasnya (BUY Re-Entry). TP "
               "saat close pertama di bawah MA5 setelah reli, hanya kalau di atas "
               "harga beli. Ini pemantauan, bukan sinyal terbukti."),
    }),
}


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

st.title(f"🏗️ {TITLE.get(lang, TITLE['en'])}")
pick = st.segmented_control(
    "STRATEGI", list(STRATEGIES), default="A",
    format_func=lambda k: f"PINESCRIPT {k}", key="strategy",
) or "A"  # clicking the active button deselects it; keep showing A
mod, table_key, caption = STRATEGIES[pick]
st.caption(caption.get(lang, caption["en"]))


# The build walks every ticker's bars through the state machine — ~22s — and the
# store only changes once a trading day, so the short radar TTL was wasteful.
def _data_version() -> str:
    """The published slim-store version, read here rather than from idxcore.

    Deliberately not `db._read_marker`: a Streamlit redeploy is often a hot
    reload that re-runs this script against already-imported modules, so any
    newly added attribute of `idxcore.store.db` is missing until a real reboot
    and the page dies with an AttributeError. Page code is always fresh.
    """
    try:
        return (Path("data") / "slim_version.txt").read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


@st.cache_data(ttl=3600, show_spinner="Menyusun tabel trade…")
def _log(version: str, pick: str):
    with _connection() as con:
        return None if con is None else tl.build(con, STRATEGIES[pick][0])


st.subheader(t("ms_screener", lang))
try:
    log = _log(_data_version(), pick)
except StoreBusy:
    st.info(t("store_busy", lang), icon="⏳")
    st.stop()

if log is None or log.empty:
    st.error(t("no_store", lang, path=DB_PATH))
    st.stop()

trade_table.render(log, lang, key=table_key)
