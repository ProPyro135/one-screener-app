"""Strategi — the owner's Pine Script strategies, one trade table.

Pick a strategy (PINESCRIPT A = Market Structure, B = Reversal Sniper, C =
IDX Adaptive Swing v4) and the
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
from idxcore.compute import swing_adaptive as sa  # noqa: E402
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
    "C": (sa, "sa", {
        "en": ("IDX Adaptive Swing v4 (end-of-day, preset \"very selective\"). The "
               "regime (uptrend, sideways, downtrend) comes from the last two swing "
               "highs and lows. BUY only at a deep swing low: the bottom 10% of the "
               "60-day range (25% in an uptrend) with low RSI and Stochastic, on a "
               "green bar closing above yesterday's high, on a stock trading at "
               "least Rp2bn a day, filled at the NEXT day's open. TP (Swing High) "
               "when Stochastic reaches 80 on a red bar 1% over the buy, sold at the "
               "next open. CL at the 5-day low minus 2.5 ATR. Monitoring, not a "
               "proven signal."),
        "id": ("IDX Adaptive Swing v4 (EOD, preset \"Sangat selektif\"). Regime "
               "(uptrend, sideways, downtrend) dibaca dari dua swing high dan swing "
               "low terakhir. BUY hanya di swing low yang dalam: 10% terbawah range "
               "60 hari (25% saat uptrend) dengan RSI dan Stochastic rendah, candle "
               "hijau close di atas high kemarin, saham dengan nilai transaksi "
               "minimal Rp2 miliar per hari; dibeli di OPEN BESOKNYA. TP (Swing High) "
               "saat Stochastic mencapai 80 dengan candle merah dan sudah untung 1%, "
               "dijual di open besoknya. CL di low 5 hari dikurangi 2,5 ATR. Ini "
               "pemantauan, bukan sinyal terbukti."),
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


# The trade log is read once per published data version; the store only
# changes once a trading day, so an hour-long cache costs nothing.
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
        if con is None:
            return None
        # The nightly publish precomputes every trade over the full ten years
        # (trade_log.publish). Reading it is instant; walking the bars here
        # would only see the slim store's ~180 bars and would lag the page.
        # The table name is spelled out rather than read from trade_log, so a
        # hot reload that keeps an older trade_log in memory cannot break this.
        cached = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'trade_log_cache'"
        ).fetchone()[0]
        if cached:
            return con.execute(
                "SELECT * EXCLUDE (strategy, seq) FROM trade_log_cache "
                "WHERE strategy = ? ORDER BY seq",
                [pick],
            ).df()
        return tl.build(con, STRATEGIES[pick][0])


try:
    log = _log(_data_version(), pick)
except StoreBusy:
    st.info(t("store_busy", lang), icon="⏳")
    st.stop()

if log is None or log.empty:
    st.error(t("no_store", lang, path=DB_PATH))
    st.stop()

trade_table.render(log, lang, key=table_key)
