"""Bilingual UI strings — English and Bahasa Indonesia.

Kept in one file so a string cannot exist in one language only. :func:`t`
raises on an unknown key rather than falling back silently, because a missing
translation that renders as a raw key is easier to spot and fix than one that
quietly shows the wrong language.

The careful wording around what is *not* measured matters as much in Indonesian
as in English, so those strings are translated in full rather than left in
English.
"""

from __future__ import annotations

import os

LANGUAGES = {"en": "English", "id": "Bahasa Indonesia"}

#: Indonesian is the default: this is an IDX tool for an Indonesian market.
#: Override with IDXCORE_LANG=en, or with the selector in the sidebar.
DEFAULT_LANG = "id"


def default_language() -> str:
    lang = os.environ.get("IDXCORE_LANG", DEFAULT_LANG).lower().strip()
    return lang if lang in LANGUAGES else DEFAULT_LANG


STRINGS: dict[str, dict[str, str]] = {
    # ---- shell ----
    "app_title": {"en": "One Screener", "id": "One Screener"},
    "app_caption": {
        "en": "IDX signal dashboard · local, read-only, free data",
        "id": "Dasbor sinyal IDX · lokal, hanya-baca, data gratis",
    },
    "language": {"en": "Language", "id": "Bahasa"},
    "theme": {"en": "Theme", "id": "Tema"},
    "theme_dark": {"en": "Dark", "id": "Gelap"},
    "theme_light": {"en": "Light", "id": "Terang"},
    "tab_screener": {"en": "Buy Recommendations & Chart", "id": "Rekomendasi Beli & Grafik"},
    "tab_guide": {"en": "📖 Guide — what the columns mean",
                  "id": "📖 Panduan — arti tiap kolom"},
    "tab_health": {"en": "Data health", "id": "Kesehatan data"},
    "guide_heading": {"en": "Reading this screener",
                      "id": "Cara membaca screener ini"},
    "guide_intro": {
        "en": ("Everything on the Buy Recommendations tab, explained in plain "
               "language. Nothing here needs prior knowledge of technical analysis."),
        "id": ("Semua yang ada di tab Rekomendasi Beli, dijelaskan dengan bahasa "
               "sehari-hari. Tidak perlu paham analisis teknikal untuk membacanya."),
    },
    "legend_inline": {
        "en": ("**Kriteria:** K3 = MA5>MA20>MA50 · K2 = + MA50>MA100 · "
               "K1 = + MA100>MA200 · K4/K5/K6 = the same three, plus RSI > 50. "
               "They nest, so a perfect setup shows all six. "
               "Full explanation in the **Guide** tab."),
        "id": ("**Kriteria:** K3 = MA5>MA20>MA50 · K2 = + MA50>MA100 · "
               "K1 = + MA100>MA200 · K4/K5/K6 = tiga yang sama, ditambah RSI > 50. "
               "Sifatnya bertingkat, jadi setup sempurna menampilkan keenamnya. "
               "Penjelasan lengkap ada di tab **Panduan**."),
    },

    # ---- freshness ----
    "data_as_of": {"en": "Data as of {date} ({days} calendar days ago)",
                   "id": "Data per {date} ({days} hari kalender lalu)"},
    "no_data_yet": {"en": "No price data in the store yet.",
                    "id": "Belum ada data harga di penyimpanan."},
    "note_aging": {"en": " — worth running `idxcore daily`.",
                   "id": " — sebaiknya jalankan `idxcore daily`."},
    "note_stale": {
        "en": " — this is old. Run `idxcore daily` before trusting anything below.",
        "id": " — ini sudah lama. Jalankan `idxcore daily` sebelum mempercayai apa pun di bawah.",
    },
    "metric_universe": {"en": "Tickers in universe", "id": "Saham terdaftar"},
    "metric_active": {"en": "Active", "id": "Aktif"},
    "metric_active_help": {
        "en": "Delisted names keep their history and stay in backtests",
        "id": "Saham delisting tetap menyimpan riwayatnya dan tetap masuk backtest",
    },
    "metric_with_data": {"en": "With price data", "id": "Punya data harga"},
    "metric_at_latest": {"en": "Reporting on latest date", "id": "Ada di tanggal terakhir"},
    "metric_at_latest_help": {
        "en": "Fewer than the universe is normal — suspensions and young listings",
        "id": "Wajar kalau lebih sedikit — ada suspensi dan saham yang baru listing",
    },
    "last_run": {"en": "Last ingest run `{run}` — {detail}.",
                 "id": "Proses ambil data terakhir `{run}` — {detail}."},
    "last_run_failures": {
        "en": ("⚠️ Last ingest run `{run}` — {detail}. {n} ticker(s) returned no data; "
               "they are logged in `ingest_log`, not silently dropped."),
        "id": ("⚠️ Proses ambil data terakhir `{run}` — {detail}. {n} saham tidak "
               "mengembalikan data; semuanya tercatat di `ingest_log`, tidak dibuang diam-diam."),
    },

    # ---- screener ----
    "screener_heading": {"en": "Buy Recommendations", "id": "Rekomendasi Beli"},
    "screener_caption": {
        "en": ("Ranked by alignment depth, then RSI. The six *Kriteria* are rendered from "
               "two stored values — alignment depth and the RSI gate — because that is what "
               "they actually are. **The chart is below this table.**"),
        "id": ("Diurutkan berdasarkan kedalaman alignment, lalu RSI. Enam *Kriteria* "
               "ditampilkan dari dua nilai tersimpan — kedalaman alignment dan gate RSI — "
               "karena itulah wujud sebenarnya. **Grafik ada di bawah tabel ini.**"),
    },
    "filters": {"en": "Filters", "id": "Filter"},
    "f_min_depth": {"en": "Minimum alignment depth", "id": "Kedalaman alignment minimum"},
    "f_min_depth_help": {
        "en": "5 = MA5>20>50>100>200. 0 = no ordered stack.",
        "id": "5 = MA5>20>50>100>200. 0 = tidak ada susunan yang urut.",
    },
    "f_gate": {"en": "Require RSI > 50", "id": "Wajib RSI > 50"},
    "f_above_ma5": {"en": "Require close above MA5", "id": "Wajib close di atas MA5"},
    "f_above_ma5_help": {
        "en": ("Not a same-day crossing. Requiring the cross to happen today excluded "
               "established trends entirely — see 'days since cross'."),
        "id": ("Bukan cross di hari yang sama. Mewajibkan cross terjadi hari ini justru "
               "membuang tren yang sudah mapan — lihat kolom 'hari sejak cross'."),
    },
    "f_max_days": {"en": "Max days since MA5 cross", "id": "Maks hari sejak cross MA5"},
    "f_max_days_help": {
        "en": "Lower this to find fresher entries. 250 keeps everything.",
        "id": "Turunkan untuk mencari entry yang lebih baru. 250 berarti semua ikut.",
    },
    "f_hide_unclean": {"en": "Hide stale / suspect bars",
                       "id": "Sembunyikan bar stale / mencurigakan"},
    "f_show_short": {"en": "Include short-history names",
                     "id": "Ikutkan saham berriwayat pendek"},
    "f_show_short_help": {
        "en": "Under 200 bars, so MA200 does not exist. Marked separately.",
        "id": "Di bawah 200 bar, jadi MA200 belum ada. Ditandai terpisah.",
    },
    "f_only_active": {"en": "Active listings only", "id": "Hanya saham aktif"},
    "match_count": {"en": "**{n}** of {total} tickers match.",
                    "id": "**{n}** dari {total} saham cocok."},
    "no_match": {
        "en": ("Nothing matches. With a small universe this is normal — loosen the "
               "filters, or load the full IDX list with `idxcore sync-universe --from-file`."),
        "id": ("Tidak ada yang cocok. Wajar kalau jumlah sahamnya masih sedikit — "
               "longgarkan filter, atau muat daftar IDX lengkap dengan "
               "`idxcore sync-universe --from-file`."),
    },

    # ---- table columns ----
    "c_ticker": {"en": "Ticker", "id": "Kode"},
    "c_name": {"en": "Name", "id": "Nama"},
    "c_close": {"en": "Close", "id": "Close"},
    "c_tier": {"en": "Tier", "id": "Tier"},
    "c_depth": {"en": "Depth", "id": "Kedalaman"},
    "c_depth_help": {"en": "3, 4 or 5 — how deep the MA stack is ordered",
                     "id": "3, 4 atau 5 — sedalam apa susunan MA yang urut"},
    "c_rsi": {"en": "RSI(14)", "id": "RSI(14)"},
    "c_gate": {"en": "RSI>50", "id": "RSI>50"},
    "c_days": {"en": "Days since MA5 cross", "id": "Hari sejak cross MA5"},
    "c_days_help": {"en": "0 = crossed today. Blank = below MA5.",
                    "id": "0 = cross hari ini. Kosong = di bawah MA5."},
    "c_criteria": {"en": "Kriteria", "id": "Kriteria"},
    "c_quality": {"en": "Data", "id": "Kualitas"},
    "c_bars": {"en": "Bars", "id": "Bar"},

    # ---- quality labels ----
    "q_ok": {"en": "OK", "id": "OK"},
    "q_short": {"en": "Short history", "id": "Riwayat pendek"},
    "q_stale": {"en": "Stale price", "id": "Harga mandek"},
    "q_suspect": {"en": "Suspect bar", "id": "Bar mencurigakan"},

    # ---- plain-language explanations ----
    "explain_expander": {
        "en": "What do these columns mean?",
        "id": "Apa arti kolom-kolom ini?",
    },
    "explain_criteria_head": {"en": "Kriteria K1-K6", "id": "Kriteria K1-K6"},
    "explain_criteria_body": {
        "en": """These are **not six separate tests**. They are one question asked at three
depths, with and without an RSI check.

The MA stack is just the moving averages in order, fastest on top. The deeper
the stack holds, the longer the uptrend has been intact.

| | Needs | Meaning |
|---|---|---|
| **K3** | MA5 > MA20 > MA50 | short-term uptrend |
| **K2** | + MA50 > MA100 | medium-term too |
| **K1** | + MA100 > MA200 | long-term as well |
| **K6** | K3 **and** RSI > 50 | short-term, with momentum |
| **K5** | K2 **and** RSI > 50 | medium-term, with momentum |
| **K4** | K1 **and** RSI > 50 | the full stack, with momentum |

They stack up. A stock showing K1 automatically shows K2 and K3 as well,
because a five-deep stack contains the three-deep one inside it. A perfect
setup lights all six at once — which is why the table stores **depth** and
**RSI gate** instead, and renders these labels from those two.""",
        "id": """Ini **bukan enam tes terpisah**. Ini satu pertanyaan yang sama, ditanyakan
pada tiga kedalaman, dengan dan tanpa cek RSI.

Susunan MA itu sekadar moving average yang berurutan, yang tercepat di atas.
Makin dalam susunannya bertahan, makin lama tren naiknya sudah berjalan.

| | Syarat | Artinya |
|---|---|---|
| **K3** | MA5 > MA20 > MA50 | tren naik jangka pendek |
| **K2** | + MA50 > MA100 | jangka menengah juga |
| **K1** | + MA100 > MA200 | jangka panjang juga |
| **K6** | K3 **dan** RSI > 50 | jangka pendek, ada momentum |
| **K5** | K2 **dan** RSI > 50 | jangka menengah, ada momentum |
| **K4** | K1 **dan** RSI > 50 | susunan penuh, ada momentum |

Sifatnya bertingkat. Saham yang kena K1 otomatis kena K2 dan K3 juga, karena
susunan lima tingkat sudah memuat yang tiga tingkat di dalamnya. Setup sempurna
menyalakan keenamnya sekaligus — itu sebabnya tabel ini menyimpan **kedalaman**
dan **gate RSI** saja, lalu label-label ini ditampilkan dari keduanya.""",
    },
    "explain_days_head": {
        "en": "Days since MA5 cross", "id": "Hari sejak cross MA5",
    },
    "explain_days_body": {
        "en": """**How many trading days ago the price rose back above its MA5 line.**

MA5 is the average close of the last 5 trading days. When price crosses from
below that line to above it, that is the moment short-term buying took over.
This column counts the sessions since that moment.

* **0** — it crossed today. The freshest possible entry.
* **7** — it crossed seven trading days ago and has stayed above since.
* **blank** — price is currently *below* MA5, so there is nothing to count.

Holidays and weekends are not counted, because they are not trading days.

**Why it exists.** The original screener only showed stocks that crossed
*today* and hid everything else. Measured on this data, that discarded **78.7%**
of otherwise-valid setups — including the strongest trends, which had simply
crossed earlier. Now nothing is hidden: sort by this column to find fresh
entries, or ignore it to see established trends.""",
        "id": """**Sudah berapa hari bursa sejak harga naik kembali di atas garis MA5-nya.**

MA5 adalah rata-rata harga penutupan 5 hari bursa terakhir. Saat harga menembus
dari bawah garis itu ke atasnya, di situlah pembeli jangka pendek mulai
menguasai. Kolom ini menghitung sudah berapa sesi berlalu sejak saat itu.

* **0** — cross-nya hari ini. Entry paling baru.
* **7** — cross tujuh hari bursa lalu, dan sejak itu bertahan di atas.
* **kosong** — harga sekarang *di bawah* MA5, jadi tidak ada yang dihitung.

Hari libur dan akhir pekan tidak ikut dihitung, karena bukan hari bursa.

**Kenapa ada.** Screener versi lama hanya menampilkan saham yang cross *hari itu
juga*, sisanya disembunyikan. Diukur dari data ini, cara itu membuang **78,7%**
setup yang sebenarnya valid — termasuk tren-tren terkuat, yang cross-nya
kebetulan lebih awal. Sekarang tidak ada yang disembunyikan: urutkan kolom ini
untuk mencari entry baru, atau abaikan untuk melihat tren yang sudah mapan.""",
    },
    "explain_depth_head": {"en": "Depth and Tier", "id": "Kedalaman dan Tier"},
    "explain_depth_body": {
        "en": """**Depth** is how far down the MA stack stays in order — 3, 4 or 5.

**Tier** combines depth with the RSI gate into a single grade: A is a five-deep
stack with RSI above 50; D is three-deep with RSI below it.

A tier describes trend *structure*. It is not a probability, and on its own it
predicts nothing.""",
        "id": """**Kedalaman** = sampai tingkat berapa susunan MA tetap urut — 3, 4, atau 5.

**Tier** menggabungkan kedalaman dengan gate RSI jadi satu nilai: A berarti
susunan lima tingkat dengan RSI di atas 50; D berarti tiga tingkat dengan RSI
di bawahnya.

Tier menggambarkan *struktur* tren. Bukan probabilitas, dan sendirian tidak
memprediksi apa pun.""",
    },
    "explain_colours_head": {"en": "Chart colours", "id": "Warna di grafik"},
    "explain_colours_body": {
        "en": """MA5 white · MA20 orange · MA50 blue · MA100 red · MA200 violet.

The chart panel is dark so all five stay legible, and the x-axis skips weekends
and exchange holidays — only real trading days are drawn, so there are no flat
gaps across days the market was shut.

On the RSI panel, the amber dashed line is the **50 gate** that criteria K4, K5
and K6 depend on. The faint dotted lines at 70 and 30 are the conventional
overbought/oversold marks; this system does not use them.""",
        "id": """MA5 putih · MA20 oranye · MA50 biru · MA100 merah · MA200 ungu.

Latar grafik dibuat gelap supaya kelimanya tetap terbaca, dan sumbu x melewati
akhir pekan serta hari libur bursa — yang digambar hanya hari bursa sungguhan,
jadi tidak ada celah datar di hari-hari pasar tutup.

Di panel RSI, garis putus-putus kuning adalah **gate 50**, yang jadi syarat
kriteria K4, K5, dan K6. Garis titik-titik samar di 70 dan 30 itu tanda
overbought/oversold konvensional; sistem ini tidak memakainya.""",
    },

    # ---- measured outcomes (only shown once a backtest exists) ----
    "measured_heading": {
        "en": "What these tiers did historically",
        "id": "Hasil historis tiap tier",
    },
    "measured_intro": {
        "en": ("Measured over **{trades:,} trades**, {first} to {last}. Every "
               "figure traces to the calibration report and can be reproduced "
               "from the store."),
        "id": ("Diukur dari **{trades:,} transaksi**, {first} sampai {last}. "
               "Semua angka bisa dilacak ke laporan kalibrasi dan dihasilkan "
               "ulang dari penyimpanan."),
    },
    "m_tier": {"en": "Tier", "id": "Tier"},
    "m_trades": {"en": "Trades", "id": "Transaksi"},
    "m_win": {"en": "Win %", "id": "Menang %"},
    "m_mean": {"en": "Mean %", "id": "Rata-rata %"},
    "m_median": {"en": "Median %", "id": "Median %"},
    "m_mae": {"en": "Median drawdown", "id": "Median penurunan"},
    "measured_caveats": {
        "en": ("**This is history, not a forecast.** Three things it does not "
               "say:\n\n"
               "- The **median is negative** in every tier. More than half of "
               "trades lost money; a few large winners produced the average.\n"
               "- **Fees are assumed**, not taken from your broker. Change them "
               "in `config/costs.json` and the numbers shift.\n"
               "- **2026 is negative so far** — the only losing year in eleven. "
               "The most recent data is the least flattering."),
        "id": ("**Ini riwayat, bukan ramalan.** Tiga hal yang tidak dikatakannya:\n\n"
               "- **Median negatif di semua tier.** Lebih dari separuh transaksi "
               "rugi; yang mengangkat rata-rata adalah sedikit pemenang besar.\n"
               "- **Biaya masih asumsi**, bukan dari brokermu. Ubah di "
               "`config/costs.json` dan angkanya bergeser.\n"
               "- **2026 sejauh ini negatif** — satu-satunya tahun rugi dari "
               "sebelas. Data terbaru justru yang paling tidak menyenangkan."),
    },
    "measured_rule": {
        "en": "Outcome rule: `{rule}` — held to the horizon, no target, no stop.",
        "id": "Aturan hasil: `{rule}` — ditahan sampai horizon, tanpa target, tanpa stop.",
    },
    "not_this_stock": {
        "en": ("These are averages across the whole tier. They say nothing about "
               "any individual stock in the table above."),
        "id": ("Ini rata-rata seluruh tier. Sama sekali tidak mengatakan apa pun "
               "tentang saham tertentu di tabel di atas."),
    },

    # ---- the honesty notice ----
    "no_winrate": {
        "en": ("**No win rate is shown here, because none has been measured yet.** "
               "Whether these criteria predict anything is the backtest's job "
               "(IVI-77 → IVI-81). Run `idxcore backtest`, and the measured "
               "outcomes appear below this table. Until then this says what the "
               "criteria *say*, not what they are *worth*."),
        "id": ("**Tidak ada win rate di sini, karena memang belum ada yang diukur.** "
               "Apakah kriteria ini bisa memprediksi sesuatu adalah tugas backtest "
               "(IVI-77 → IVI-81). Jalankan `idxcore backtest`, dan hasil "
               "terukurnya akan muncul di bawah tabel ini. Sampai saat itu, ini "
               "hanya menyatakan *apa kata* kriteria, bukan *seberapa berharga*."),
    },
    "tiers_expander": {"en": "What the tiers mean", "id": "Arti tiap tier"},
    "tier_a": {"en": "MA5>20>50>100>200 and RSI above 50",
               "id": "MA5>20>50>100>200 dan RSI di atas 50"},
    "tier_b": {"en": "Four-deep alignment with RSI above 50, or full alignment without it",
               "id": "Alignment 4 tingkat dengan RSI di atas 50, atau alignment penuh tanpa itu"},
    "tier_c": {"en": "Three-deep alignment with RSI above 50, or four-deep without it",
               "id": "Alignment 3 tingkat dengan RSI di atas 50, atau 4 tingkat tanpa itu"},
    "tier_d": {"en": "Three-deep alignment, RSI below 50",
               "id": "Alignment 3 tingkat, RSI di bawah 50"},
    "tier_note": {
        "en": ("Tiers grade trend structure, not probability. They are derived from the "
               "two stored axes and carry no claim about outcomes."),
        "id": ("Tier menilai struktur tren, bukan probabilitas. Tier diturunkan dari dua "
               "sumbu tersimpan dan tidak mengklaim apa pun soal hasil."),
    },

    # ---- chart ----
    "chart_heading": {"en": "Chart", "id": "Grafik"},
    "chart_ticker": {"en": "Ticker", "id": "Saham"},
    "chart_bars": {"en": "Bars to show", "id": "Jumlah bar ditampilkan"},
    "chart_only_matches": {"en": "Limit to matching tickers",
                           "id": "Batasi ke saham yang cocok"},
    "chart_show_extras": {"en": "Show moving averages & RSI",
                          "id": "Tampilkan moving average & RSI"},
    "chart_extras_help": {
        "en": "Off by default, to match the TradingView-style chart. Turn on to "
              "see the five MAs and the RSI panel behind the six criteria.",
        "id": "Mati secara default, agar mirip grafik ala TradingView. Nyalakan "
              "untuk melihat lima MA dan panel RSI di balik enam kriteria.",
    },
    "chart_no_history": {"en": "No stored history for this ticker.",
                         "id": "Belum ada riwayat tersimpan untuk saham ini."},
    "chart_price_note": {
        "en": ("Prices are raw, exactly as the exchange printed them — not "
               "dividend-adjusted — so they match a broker screen."),
        "id": ("Harga mentah, persis seperti yang dicetak bursa — bukan hasil penyesuaian "
               "dividen — jadi cocok dengan layar broker."),
    },
    "chart_rsi_note": {"en": "Wilder smoothing, matching TradingView and TTR::RSI.",
                       "id": "Smoothing Wilder, sama dengan TradingView dan TTR::RSI."},
    "chart_controls": {
        "en": ("Drag to move through time · scroll to zoom · scroll on the price "
               "or date axis to stretch that axis alone · double-click to reset."),
        "id": ("Geser untuk berpindah waktu · scroll untuk zoom · scroll di sumbu "
               "harga atau tanggal untuk menarik sumbu itu saja · klik dua kali "
               "untuk kembali ke tampilan penuh."),
    },
    "chart_no_rsi": {"en": "Not enough bars yet for RSI(14).",
                     "id": "Bar belum cukup untuk menghitung RSI(14)."},
    "raw_rows": {"en": "Raw stored rows (most recent 20)",
                 "id": "Baris mentah tersimpan (20 terbaru)"},

    # ---- per-ticker warnings ----
    "warn_short": {
        "en": ("Only {bars} bars — MA200 does not exist yet. This is missing data, "
               "not a weak trend."),
        "id": ("Baru {bars} bar — MA200 belum ada. Ini data yang belum lengkap, "
               "bukan tren yang lemah."),
    },
    "warn_suspect": {
        "en": ("A recent bar moved more than 35% with no split recorded. Likely an "
               "unadjusted corporate action — treat the indicators with suspicion "
               "until it is checked."),
        "id": ("Ada bar yang bergerak lebih dari 35% tanpa catatan split. Kemungkinan "
               "aksi korporasi yang belum disesuaikan — curigai dulu indikatornya "
               "sampai diperiksa."),
    },
    "warn_stale": {
        "en": ("This price has not moved for several sessions. RSI drifts to 50 and the "
               "moving averages flatten on stale data."),
        "id": ("Harga ini tidak bergerak beberapa sesi. Pada data mandek, RSI menepi ke 50 "
               "dan moving average jadi datar."),
    },

    # ---- data health ----
    "health_heading": {"en": "Data health", "id": "Kesehatan data"},
    "health_caption": {
        "en": ("Every check flags a bar; none delete one. Dropping a bad bar would "
               "rewrite history silently."),
        "id": ("Setiap pemeriksaan menandai bar, tidak ada yang menghapus. Membuang bar "
               "yang buruk sama saja menulis ulang sejarah diam-diam."),
    },
    "health_none": {"en": "No quality flags computed yet. Run `idxcore compute`.",
                    "id": "Belum ada tanda kualitas yang dihitung. Jalankan `idxcore compute`."},
    "health_perday": {"en": "**Qualifying tickers per day**",
                      "id": "**Jumlah saham yang lolos per hari**"},
    "health_perday_note": {
        "en": "A line flat at zero means nothing qualified — a real answer, not a bug.",
        "id": "Garis yang datar di nol berarti tidak ada yang lolos — itu jawaban nyata, bukan bug.",
    },

    # ---- empty states ----
    "no_store": {"en": "No store at `{path}`.", "id": "Tidak ada penyimpanan di `{path}`."},
    "store_busy": {
        "en": "The nightly sync is updating the store right now, and it takes "
              "about three minutes. The store allows one writer or many "
              "readers, never both, so the dashboard has to wait its turn. "
              "Nothing is wrong and nothing is lost.",
        "id": "Sinkronisasi harian sedang memperbarui penyimpanan, dan itu "
              "butuh sekitar tiga menit. Penyimpanan hanya mengizinkan satu "
              "penulis atau banyak pembaca, tidak keduanya, jadi dasbor harus "
              "menunggu giliran. Tidak ada yang salah dan tidak ada yang hilang.",
    },
    "store_busy_retry": {"en": "Try again", "id": "Coba lagi"},
    "no_store_help": {"en": "Create it first:", "id": "Buat dulu:"},
    "no_signals": {
        "en": "No signals stored yet. Run `idxcore compute` after a backfill.",
        "id": "Belum ada sinyal tersimpan. Jalankan `idxcore compute` setelah backfill.",
    },

    # ---- search page ----
    "sr_title": {"en": "Search a saham", "id": "Cari saham"},
    "sr_caption": {
        "en": ("Type a code or name to look up any stock — its chart, indicators and "
               "the criteria it currently meets. Read-only, like the rest of the app."),
        "id": ("Ketik kode atau nama untuk mencari saham mana pun — grafik, indikator, "
               "dan kriteria yang sedang dipenuhinya. Hanya-baca, seperti bagian lain."),
    },
    "sr_saham": {"en": "Saham", "id": "Saham"},
    "sr_placeholder": {"en": "e.g. BBCA, or Telkom", "id": "mis. BBCA, atau Telkom"},
    "sr_meta": {
        "en": "Criteria: {crit}  ·  Board: {board}  ·  Sector: {sector}",
        "id": "Kriteria: {crit}  ·  Papan: {board}  ·  Sektor: {sector}",
    },

    # ---- search-page radar screener ----
    "sr_screener": {"en": "Radar — where each stock is now",
                    "id": "Radar — posisi tiap saham sekarang"},
    "sr_screener_count": {"en": "**{n}** stock(s) shown.", "id": "**{n}** saham ditampilkan."},
    "sr_status_filter": {"en": "Filter by status", "id": "Saring berdasarkan status"},
    "sr_status": {"en": "Status", "id": "Status"},
    "sr_signal": {"en": "Latest signal", "id": "Sinyal terakhir"},
    "sr_reason": {"en": "Reason", "id": "Alasan"},
    "sr_lookup": {"en": "Look up one stock", "id": "Cari satu saham"},
    "st_naik": {"en": "RISING", "id": "LAGI NAIK"},
    "st_bottom": {"en": "WATCH (BOTTOM)", "id": "PANTAU (BOTTOM)"},
    "st_tunggu": {"en": "WAIT", "id": "TUNGGU"},
    "st_pantau": {"en": "WATCH (PULLBACK)", "id": "PANTAU (PULLBACK)"},
    "ms_title": {"en": "Market Structure", "id": "Market Structure"},
    "ms_caption": {
        "en": ("A second screener. Auto-switching regime (EMA20 in a volatile "
               "swing, else SMA200), a rebound off a higher swing-low on dry "
               "volume, then a volume-backed breakout. A monitoring view, not a "
               "proven signal."),
        "id": ("Screener kedua. Regime otomatis (EMA20 saat swing volatil, "
               "selain itu SMA200), rebound dari swing-low yang lebih tinggi "
               "dengan volume kering, lalu breakout didukung volume. Ini "
               "pemantauan, bukan sinyal terbukti."),
    },
    "ms_screener": {"en": "Radar — where each stock is now",
                    "id": "Radar — posisi tiap saham sekarang"},
    "ms_signal_filter": {"en": "Filter by signal", "id": "Saring berdasarkan sinyal"},
    "ms_no_signal": {"en": "No signal", "id": "Tanpa sinyal"},
}


def t(key: str, lang: str = DEFAULT_LANG, **kwargs) -> str:
    """Look up a string. Raises on an unknown key rather than guessing."""
    try:
        entry = STRINGS[key]
    except KeyError as exc:
        raise KeyError(f"no UI string named {key!r}") from exc
    text = entry.get(lang) or entry[DEFAULT_LANG]
    return text.format(**kwargs) if kwargs else text


def missing_translations() -> dict[str, list[str]]:
    """Keys that lack a translation in some language. Used by the tests."""
    gaps: dict[str, list[str]] = {}
    for key, entry in STRINGS.items():
        absent = [lang for lang in LANGUAGES if not entry.get(lang)]
        if absent:
            gaps[key] = absent
    return gaps
