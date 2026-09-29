"""Strategi — the owner's Pine Script strategies, one trade table.

Pick a strategy (PINESCRIPT A = Market Structure, B = Reversal Sniper, C =
Pattern Breakout VCP & Double Bottom, D = Accumulation Breakout, E = Advanced
Breakout, F = Early Entry & Hard TP, G = Uptrend Buy The Dip, H = G with its IHSG and ADX
filters, I = Sniper VCP, J = Advanced Breakout v2) and the
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
from idxcore.compute import accumulation_breakout as ab  # noqa: E402
from idxcore.compute import advanced_breakout as adv  # noqa: E402
from idxcore.compute import advanced_breakout_v2 as adv2  # noqa: E402
from idxcore.compute import buy_the_dip as btd  # noqa: E402
from idxcore.compute import buy_the_dip_filtered as btdf  # noqa: E402
from idxcore.compute import early_entry as ee  # noqa: E402
from idxcore.compute import sniper_vcp as sv  # noqa: E402
from idxcore.compute import market_structure as ms  # noqa: E402
from idxcore.compute import pattern_breakout as pb  # noqa: E402
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
    "C": (pb, "pb", {
        "en": ("Pattern Breakout: VCP and Double Bottom. VCP: a base of 2+ ever "
               "shallower pullbacks (first at most 35%, last at most 10%) in a "
               "Minervini uptrend, with volume drying up. Double Bottom: two swing "
               "lows within 4% of each other under a neckline. BUY on a close "
               "through the pivot or neckline (at most 5% over it) on 1.5x volume, "
               "on a stock trading at least Rp5bn a day, filled at the NEXT day's "
               "open. Stop at the pattern low, at most 8% under. TP: half the "
               "position at 2x the risk, then the stop moves to the buy price and "
               "the rest is sold after a close under EMA21; P&L is the average of "
               "both halves. CL: the stop. Monitoring, not a proven signal."),
        "id": ("Pattern Breakout: VCP dan Double Bottom. VCP: base dengan 2+ koreksi "
               "yang makin dangkal (pertama maks 35%, terakhir maks 10%) dalam "
               "uptrend Minervini, volume mengering. Double Bottom: dua swing low "
               "setara (selisih maks 4%) di bawah neckline. BUY saat close menembus "
               "pivot atau neckline (maks 5% di atasnya) dengan volume 1,5x, saham "
               "dengan nilai transaksi minimal Rp5 miliar per hari; dibeli di OPEN "
               "BESOKNYA. Stop di low pola, maks 8% di bawah. TP: separuh posisi "
               "dijual di 2x risiko, lalu stop digeser ke harga beli dan sisanya "
               "dijual setelah close di bawah EMA21; P&L adalah rata-rata kedua "
               "bagian. CL: kena stop. Ini pemantauan, bukan sinyal terbukti."),
    }),
    "D": (ab, "ab", {
        "en": ("Accumulation Breakout & Trend Run. Accumulation: the last 25 days "
               "moved within a 15% range, near or above EMA50. BUY on a close "
               "through that range's high on 1.3x volume, filled at the NEXT day's "
               "open. No fixed target: a trailing stop (highest high since the buy "
               "minus 3 ATR) follows the run, and a close under it sells at the "
               "next open. Sold above the buy price it counts as TP, otherwise as "
               "CL. Monitoring, not a proven signal."),
        "id": ("Accumulation Breakout & Trend Run. Akumulasi: 25 hari terakhir "
               "bergerak dalam range maks 15%, dekat atau di atas EMA50. BUY saat "
               "close menembus high range itu dengan volume 1,3x; dibeli di OPEN "
               "BESOKNYA. Tanpa target tetap: trailing stop (high tertinggi sejak "
               "beli dikurangi 3 ATR) mengikuti kenaikan, dan close di bawahnya "
               "dijual di open besoknya. Terjual di atas harga beli dihitung TP, "
               "selain itu CL. Ini pemantauan, bukan sinyal terbukti."),
    }),
    "E": (adv, "adv", {
        "en": ("Advanced Breakout, a stricter D. Base: the last 20 days moved within "
               "a 12% range, above EMA200. BUY on a close through that range's high "
               "on 1.5x the 50-day volume, closing in the top 30% of its candle, "
               "filled at the NEXT day's open. The stop starts at the higher of the "
               "close minus 4 ATR and the base low, then trails the highest high "
               "since the buy minus 4 ATR; a close under it sells at the next open. "
               "Sold above the buy price it counts as TP, otherwise as CL. "
               "Monitoring, not a proven signal."),
        "id": ("Advanced Breakout, versi D yang lebih ketat. Base: 20 hari terakhir "
               "bergerak dalam range maks 12%, di atas EMA200. BUY saat close menembus "
               "high range itu dengan volume 1,5x rata-rata 50 hari dan close di 30% "
               "teratas candle; dibeli di OPEN BESOKNYA. Stop awal: yang lebih tinggi "
               "dari close dikurangi 4 ATR dan low base, lalu mengikuti high tertinggi "
               "sejak beli dikurangi 4 ATR; close di bawahnya dijual di open "
               "besoknya. Terjual di atas harga beli dihitung TP, selain itu CL. Ini "
               "pemantauan, bukan sinyal terbukti."),
    }),
    "F": (ee, "ee", {
        "en": ("Early Entry & Hard TP. Base: the last 20 days moved within a 15% "
               "range, above EMA100. BUY early, on a close through the 10-day high "
               "on 1.5x the 50-day volume, closing in the top 40% of its candle, "
               "filled at the NEXT day's open. TP: the whole position at +10% over "
               "the buy. Stop: the highest high since the buy minus 3 ATR, only "
               "rising. Both work intraday from the day after the buy; when a day "
               "touches both, the stop counts. A stop above the buy price counts "
               "as TP. Monitoring, not a proven signal."),
        "id": ("Early Entry & Hard TP. Base: 20 hari terakhir bergerak dalam range "
               "maks 15%, di atas EMA100. BUY lebih awal, saat close menembus high "
               "10 hari dengan volume 1,5x rata-rata 50 hari dan close di 40% "
               "teratas candle; dibeli di OPEN BESOKNYA. TP: seluruh posisi di +10% "
               "dari harga beli. Stop: high tertinggi sejak beli dikurangi 3 ATR, "
               "hanya naik. Keduanya berlaku intraday mulai sehari setelah beli; "
               "kalau satu hari menyentuh keduanya, dihitung kena stop. Stop di "
               "atas harga beli dihitung TP. Ini pemantauan, bukan sinyal terbukti."),
    }),
    "G": (btd, "btd", {
        "en": ("Uptrend Buy The Dip (MA20 > MA50). BUY on a golden cross (MA20 "
               "crossing over MA50), then while MA20 stays over MA50, BUY again on "
               "every dip: the low touches MA20, the close stays over MA50, on a "
               "green candle. Up to 5 buys stacked, at least 5 days apart; each buy "
               "is its own row. All filled at the NEXT day's open. No fixed target: "
               "everything is sold at the next open after a close under the "
               "trailing stop (close minus 3 ATR, only rising) or MA20 crossing "
               "under MA50. A row sold above its buy price counts as TP, otherwise "
               "as CL. Monitoring, not a proven signal."),
        "id": ("Uptrend Buy The Dip (MA20 > MA50). BUY saat golden cross (MA20 "
               "memotong ke atas MA50), lalu selama MA20 di atas MA50, BUY lagi di "
               "setiap dip: low menyentuh MA20, close tetap di atas MA50, candle "
               "hijau. Maks 5 pembelian bertumpuk, jarak minimal 5 hari; tiap "
               "pembelian satu baris. Semua dibeli di OPEN BESOKNYA. Tanpa target "
               "tetap: semua dijual di open besoknya setelah close di bawah trailing "
               "stop (close dikurangi 3 ATR, hanya naik) atau MA20 memotong ke bawah "
               "MA50. Baris yang terjual di atas harga belinya dihitung TP, selain "
               "itu CL. Ini pemantauan, bukan sinyal terbukti."),
    }),
    "H": (btdf, "btdf", {
        "en": ("Uptrend Buy The Dip v2: G with two of its filters on. A BUY is only "
               "taken while IHSG is rising (IHSG MA20 over MA50 and IHSG above its "
               "MA200) and the stock's own trend is strong (ADX over 20 with DI+ "
               "over DI-). Everything else, the exits included, is as in G. "
               "Monitoring, not a proven signal."),
        "id": ("Uptrend Buy The Dip v2: G dengan dua filternya aktif. BUY hanya "
               "diambil saat IHSG sedang naik (MA20 IHSG di atas MA50 dan IHSG di "
               "atas MA200) dan tren sahamnya sendiri kuat (ADX di atas 20 dengan "
               "DI+ di atas DI-). Selebihnya, termasuk cara jual, sama dengan G. Ini "
               "pemantauan, bukan sinyal terbukti."),
    }),
    "I": (sv, "sv", {
        "en": ("Sniper VCP. Base: the last 40 days moved within a 12% range, in a "
               "Minervini stage 2 uptrend (close over EMA50 over EMA150 over EMA200). "
               "BUY on a close through the 20-day high on 2x the 50-day volume, "
               "closing in the top 30% of its candle, filled at the NEXT day's open. "
               "First stop: the higher of the base low and the signal close minus "
               "5.5%. Once the high since the buy reaches +5%, a trailing stop 3% "
               "under that high switches on (only rising). TP: the whole position at "
               "+20%. Both work intraday from the day after the buy; a day touching "
               "both counts as the stop. If the trailing never switched on, it sells "
               "10 days after the signal (time stop). As written, the Pine never "
               "sells (its entry price is wiped on the signal day), so this follows "
               "its comments, with the buy price taken as the real fill. Monitoring, "
               "not a proven signal."),
        "id": ("Sniper VCP. Base: 40 hari terakhir bergerak dalam range maks 12%, dalam "
               "uptrend stage 2 Minervini (close di atas EMA50 di atas EMA150 di atas "
               "EMA200). BUY saat close menembus high 20 hari dengan volume 2x "
               "rata-rata 50 hari dan close di 30% teratas candle; dibeli di OPEN "
               "BESOKNYA. Stop awal: yang lebih tinggi dari low base dan close sinyal "
               "dikurangi 5,5%. Setelah high sejak beli mencapai +5%, trailing stop 3% "
               "di bawah high itu aktif (hanya naik). TP: seluruh posisi di +20%. "
               "Keduanya berlaku intraday mulai sehari setelah beli; kalau satu hari "
               "menyentuh keduanya, dihitung kena stop. Kalau trailing belum pernah "
               "aktif, dijual 10 hari setelah sinyal (time stop). Pine aslinya tidak "
               "pernah menjual (harga belinya terhapus di hari sinyal), jadi ini "
               "mengikuti maksud komentarnya, dengan harga beli = harga isi "
               "sebenarnya. Ini pemantauan, bukan sinyal terbukti."),
    }),
    "J": (adv2, "adv2", {
        "en": ("Advanced Breakout v2: E's BUY with extra protection. First stop: the "
               "highest of close minus 4 ATR, the base low, and close minus 7%; then "
               "a 4 ATR trailing stop. Once up 8%, the stop is at least the buy "
               "price +1%. Within the first 10 days, a close more than 2% under the "
               "breakout level is a failed breakout and sells. After 15 days without "
               "ever reaching +5%, it sells (time stop). Every sale is at the next "
               "open; above the buy price it counts as TP, otherwise as CL. "
               "Monitoring, not a proven signal."),
        "id": ("Advanced Breakout v2: BUY sama dengan E, dengan proteksi tambahan. Stop "
               "awal: yang tertinggi dari close dikurangi 4 ATR, low base, dan close "
               "dikurangi 7%; lalu trailing 4 ATR. Setelah naik 8%, stop minimal harga "
               "beli +1%. Dalam 10 hari pertama, close lebih dari 2% di bawah level "
               "breakout dianggap breakout gagal dan dijual. Setelah 15 hari tanpa "
               "pernah naik 5%, dijual (time stop). Semua dijual di open besoknya; di "
               "atas harga beli dihitung TP, selain itu CL. Ini pemantauan, bukan "
               "sinyal terbukti."),
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
