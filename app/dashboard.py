"""One Screener — IDX signal dashboard.

Run it with::

    streamlit run app/dashboard.py

This app is a **reader**. It opens the DuckDB store read-only and displays what
is in it. It does not fetch from Yahoo, does not compute an indicator, and does
not invent a number.

Four defects from the original app are deliberately not reproduced here:

* no `RSI Skor` / `MACD Skor` / `Vol Skor` columns that read 100% on every row
  — a column with one distinct value carries no information and is cut;
* no risk/reward figure, in either direction, until the cost model and outcome
  definition exist (IVI-79 / IVI-80). A ratio shown backwards is worse than no
  ratio;
* no horizon label contradicting the estimated holding period — no horizon is
  claimed at all until the backtest measures one;
* no green "LIVE" badge. The header reports the date of the newest bar and what
  the last ingest run actually did.

Above all: **no percentage appears here unless it was measured.** That rule
cuts both ways. While `labels` is empty the app says plainly that nothing has
been measured; once a backtest has run, the tier outcomes appear — traced to
stored labels — together with the three things they do not say: the medians are
negative, the fees are assumed, and the most recent year is the losing one.

Saying "not measured yet" after it has been measured would be its own kind of
dishonesty, so that message is driven by the store rather than hard-coded.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxcore.charts import PLOTLY_CONFIG, build_combined_figure  # noqa: E402
from idxcore.compute.signals import render_criteria  # noqa: E402
from idxcore.i18n import LANGUAGES, default_language, t  # noqa: E402
from idxcore.store import db, read  # noqa: E402

st.set_page_config(page_title="One Screener — IDX", page_icon="📊", layout="wide")

def _resolve_db_path() -> str:
    """Full store locally, slim store on the free host.

    `$IDXCORE_DB` wins if set. Otherwise the full store is used when it is
    present (the developer's machine), and the committed slim read-only store is
    the fallback (the hosted deploy, where the 497MB store never travels).
    """
    env = os.environ.get("IDXCORE_DB")
    if env:
        return env
    full = Path(db.DEFAULT_DB_PATH)
    if full.exists():
        return str(full)
    # Hosted: no full store — fetch the slim snapshot from the Release asset.
    return str(db.ensure_slim_store())


DB_PATH = _resolve_db_path()

TIER_KEYS = {"A": "tier_a", "B": "tier_b", "C": "tier_c", "D": "tier_d"}
QUALITY_KEYS = {
    "ok": "q_ok",
    "insufficient_history": "q_short",
    "stale": "q_stale",
    "suspect": "q_suspect",
}


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------

class StoreBusy(RuntimeError):
    """The nightly sync is writing. DuckDB allows one writer or many readers."""


@contextmanager
def _connection():
    """A read-only connection held for exactly as long as the query takes.

    This used to be `@st.cache_resource`, which kept the connection alive for
    the life of the *server process* — so a dashboard that had served a single
    page held the read lock forever, and closing the browser tab did not
    release it. DuckDB allows one writer or many readers, so the nightly sync
    could not get in: registered, scheduled, and guaranteed to fail every
    evening the dashboard happened to be running.

    Opening per use costs ~17ms against a 24ms query and a 148ms page load —
    inside the measurement noise — and the page bundle below is cached for 60
    seconds anyway, so this runs at most once a minute per session. Automation
    that silently loses to an open browser tab is not worth that.

    The trade is the other way round: for the ~3 minutes a night the sync is
    writing, the reader cannot open either. That surfaces as StoreBusy and is
    rendered as a plain "sync in progress" message, which is a true statement
    about a transient condition rather than a stack trace.
    """
    if not Path(DB_PATH).exists():
        yield None
        return
    try:
        con = db.connect(DB_PATH, read_only=True)
    except (db.StoreLocked, duckdb.IOException) as exc:
        # db.connect already turns the lock case into StoreLocked, whose text
        # is written for someone at a prompt running a write command — it tells
        # you to stop the dashboard, which is not advice the dashboard can act
        # on. Re-raise as StoreBusy so the UI says the true thing instead.
        raise StoreBusy(str(exc)) from exc
    try:
        yield con
    finally:
        con.close()


@st.cache_data(ttl=60)
def _load():
    with _connection() as con:
        if con is None:
            return None
        return {
            "freshness": read.freshness(con),
            "signals": read.latest_signals(con),
            "universe": read.universe_summary(con),
            "quality": read.quality_summary(con),
            "counts": read.signal_counts_by_date(con),
            "measured": read.measured_tier_outcomes(con),
        }


def _pick_language() -> str:
    """Language selector, pinned to the top of the sidebar."""
    if "lang" not in st.session_state:
        st.session_state["lang"] = default_language()

    codes = list(LANGUAGES)
    with st.sidebar:
        choice = st.radio(
            t("language", st.session_state["lang"]),
            options=codes,
            index=codes.index(st.session_state["lang"]),
            format_func=lambda c: LANGUAGES[c],
            horizontal=True,
            key="lang_choice",
        )
    st.session_state["lang"] = choice
    return choice


THEMES = ("dark", "light")

#: Streamlit has no runtime theme API, so the page chrome is themed by CSS
#: injection. Only the main surfaces are set — background, text, sidebar,
#: header — and the chart itself is themed properly through charts.palette().
#: The colours mirror the snapshot's :root, so the live app and the shared
#: file read the same.
_THEME_CSS = {
    "dark": """
        <style>
          .stApp, [data-testid="stAppViewContainer"] { background-color: #0E1117; color: #FAFAFA; }
          [data-testid="stHeader"] { background-color: rgba(14,17,23,0.8); }
          [data-testid="stSidebar"] { background-color: #262730; }
        </style>
    """,
    "light": """
        <style>
          .stApp, [data-testid="stAppViewContainer"] { background-color: #FFFFFF; color: #131722; }
          [data-testid="stHeader"] { background-color: rgba(255,255,255,0.8); }
          [data-testid="stSidebar"] { background-color: #F0F2F6; }
        </style>
    """,
}


def _pick_theme(lang: str) -> str:
    """Dark/light selector, below the language radio."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

    labels = {"dark": t("theme_dark", lang), "light": t("theme_light", lang)}
    with st.sidebar:
        choice = st.radio(
            t("theme", lang),
            options=THEMES,
            index=THEMES.index(st.session_state["theme"]),
            format_func=lambda c: labels[c],
            horizontal=True,
            key="theme_choice",
        )
        st.divider()
    st.session_state["theme"] = choice
    st.markdown(_THEME_CSS[choice], unsafe_allow_html=True)
    return choice


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------

def render_freshness(f: read.Freshness, universe: dict, lang: str) -> None:
    banner = {
        "fresh": st.success,
        "aging": st.warning,
        "stale": st.error,
        "empty": st.error,
    }[f.state]

    if f.latest_date is None:
        headline = t("no_data_yet", lang)
    else:
        headline = t(
            "data_as_of", lang,
            date=f"{f.latest_date:%d %b %Y}", days=f.calendar_days_old,
        )

    note = ""
    if f.state == "aging":
        note = t("note_aging", lang)
    elif f.state == "stale":
        note = t("note_stale", lang)

    banner(f"**{headline}**{note}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("metric_universe", lang), universe["total"])
    c2.metric(t("metric_active", lang), universe["active"],
              help=t("metric_active_help", lang))
    c3.metric(t("metric_with_data", lang), f.tickers_with_data)
    c4.metric(t("metric_at_latest", lang), f.tickers_at_latest,
              help=t("metric_at_latest_help", lang))

    if f.last_run_id:
        detail = ", ".join(f"{k}: {n}" for k, n in sorted(f.last_run_statuses.items()))
        if f.failures:
            st.caption(t("last_run_failures", lang, run=f.last_run_id,
                         detail=detail, n=f.failures))
        else:
            st.caption(t("last_run", lang, run=f.last_run_id, detail=detail))


# ---------------------------------------------------------------------------
# screener
# ---------------------------------------------------------------------------

def render_screener(
    signals: pd.DataFrame, lang: str, measured: pd.DataFrame
) -> pd.DataFrame:
    st.subheader(t("screener_heading", lang))
    st.caption(t("screener_caption", lang))

    with st.sidebar:
        st.header(t("filters", lang))
        min_depth = st.select_slider(
            t("f_min_depth", lang), options=[0, 3, 4, 5], value=3,
            help=t("f_min_depth_help", lang),
        )
        require_gate = st.checkbox(t("f_gate", lang), value=True)
        require_above_ma5 = st.checkbox(
            t("f_above_ma5", lang), value=True, help=t("f_above_ma5_help", lang)
        )
        max_days = st.slider(
            t("f_max_days", lang), 0, 250, 250, help=t("f_max_days_help", lang)
        )
        hide_unclean = st.checkbox(t("f_hide_unclean", lang), value=True)
        show_short = st.checkbox(
            t("f_show_short", lang), value=False, help=t("f_show_short_help", lang)
        )
        only_active = st.checkbox(t("f_only_active", lang), value=True)

    df = signals.copy()
    if only_active and "is_active" in df:
        df = df[df["is_active"].fillna(True)]
    df = df[df["alignment_depth"] >= min_depth]
    if require_gate:
        df = df[df["rsi_gate"] == True]  # noqa: E712 - tri-state, None must not pass
    if require_above_ma5:
        df = df[df["above_ma5"] == True]  # noqa: E712
    if max_days < 250:
        df = df[df["days_since_ma5_cross"].fillna(9999) <= max_days]
    if hide_unclean:
        df = df[~df["data_quality"].isin(["stale", "suspect"])]
    if not show_short:
        df = df[df["data_quality"] != "insufficient_history"]

    st.write(t("match_count", lang, n=len(df), total=len(signals)))

    if df.empty:
        st.info(t("no_match", lang))
        return df

    table = pd.DataFrame(
        {
            t("c_ticker", lang): df["idx_code"].fillna(df["ticker"]),
            t("c_name", lang): df["name"],
            t("c_close", lang): df["close"],
            t("c_tier", lang): df["tier"],
            t("c_depth", lang): df["alignment_depth"],
            t("c_rsi", lang): df["rsi14"].round(1),
            t("c_gate", lang): df["rsi_gate"],
            t("c_days", lang): df["days_since_ma5_cross"],
            t("c_criteria", lang): [
                ", ".join(render_criteria(int(d), g)) or "—"
                for d, g in zip(df["alignment_depth"], df["rsi_gate"])
            ],
            t("c_quality", lang): [
                t(QUALITY_KEYS.get(q, "q_ok"), lang) for q in df["data_quality"]
            ],
            t("c_bars", lang): df["bars_available"],
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            t("c_close", lang): st.column_config.NumberColumn(format="%.0f"),
            t("c_rsi", lang): st.column_config.NumberColumn(format="%.1f"),
            t("c_gate", lang): st.column_config.CheckboxColumn(),
            t("c_depth", lang): st.column_config.NumberColumn(
                help=t("c_depth_help", lang)
            ),
            t("c_days", lang): st.column_config.NumberColumn(
                format="%d", help=t("c_days_help", lang)
            ),
        },
    )

    render_measured_outcomes(measured, lang)

    # A one-line legend that is always visible. The full guide has its own tab:
    # burying an explanation in a collapsed expander means it is not there for
    # anyone who does not already know to look.
    st.caption(t("legend_inline", lang))

    return df


def render_measured_outcomes(measured: pd.DataFrame, lang: str) -> None:
    """Show backtested tier outcomes — or say plainly that none exist yet.

    Invariant 8 forbids unmeasured percentages, not measured ones. While
    `labels` is empty this says so; once a backtest has run, the figures appear
    with the three things they do not say attached to them.
    """
    if measured is None or measured.empty:
        st.info(t("no_winrate", lang), icon="ℹ️")
        return

    total = int(measured["trades"].sum())
    st.markdown(f"#### {t('measured_heading', lang)}")
    st.markdown(
        t(
            "measured_intro", lang,
            trades=total,
            first=f"{measured['first_entry'].min():%b %Y}",
            last=f"{measured['last_entry'].max():%b %Y}",
        )
    )

    st.dataframe(
        pd.DataFrame(
            {
                t("m_tier", lang): measured["tier"],
                t("m_trades", lang): measured["trades"],
                t("m_win", lang): measured["win_pct"].round(1),
                t("m_mean", lang): measured["mean_pct"].round(2),
                t("m_median", lang): measured["median_pct"].round(2),
                t("m_mae", lang): measured["median_mae"].round(2),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(t("measured_caveats", lang), icon="⚠️")
    st.caption(t("not_this_stock", lang))
    st.caption(t("measured_rule", lang, rule=measured["rule_version"].iloc[0]))


def render_guide(lang: str) -> None:
    """The explanations, as a tab rather than a hidden expander."""
    st.subheader(t("guide_heading", lang))
    st.caption(t("guide_intro", lang))

    st.markdown(f"### {t('explain_criteria_head', lang)}")
    st.markdown(t("explain_criteria_body", lang))

    st.divider()
    st.markdown(f"### {t('explain_days_head', lang)}")
    st.markdown(t("explain_days_body", lang))

    st.divider()
    st.markdown(f"### {t('explain_depth_head', lang)}")
    st.markdown(t("explain_depth_body", lang))
    for tier, key in TIER_KEYS.items():
        st.markdown(f"- **{tier}** — {t(key, lang)}")
    st.caption(t("tier_note", lang))

    st.divider()
    st.markdown(f"### {t('explain_colours_head', lang)}")
    st.markdown(t("explain_colours_body", lang))


# ---------------------------------------------------------------------------
# per-ticker detail
# ---------------------------------------------------------------------------

def render_detail(
    signals: pd.DataFrame, matches: pd.DataFrame, lang: str, theme: str = "dark"
) -> None:
    st.divider()
    st.subheader(t("chart_heading", lang))
    if signals.empty:
        return

    # Default to the top-ranked match so the chart shows something relevant the
    # moment the page loads, rather than whatever sorts first alphabetically.
    options = sorted(signals["ticker"].dropna().unique())
    default = 0
    if not matches.empty:
        top = matches.iloc[0]["ticker"]
        if top in options:
            default = options.index(top)

    only_matches = st.checkbox(
        t("chart_only_matches", lang), value=False, key="detail_only_matches"
    )
    show_extras = st.checkbox(
        t("chart_show_extras", lang), value=False, key="detail_extras",
        help=t("chart_extras_help", lang),
    )
    if only_matches and not matches.empty:
        options = sorted(matches["ticker"].dropna().unique())
        default = 0

    choice = st.selectbox(
        t("chart_ticker", lang), options, index=default, key="detail_ticker"
    )
    bars = st.slider(t("chart_bars", lang), 60, 1000, 250, step=10)

    # The sync can start between the page load and this query, so the chart
    # gets the same honest message rather than an exception halfway down a page
    # that has already rendered.
    try:
        with _connection() as con:
            if con is None:
                return
            history = read.ticker_history(con, choice, limit=bars)
    except StoreBusy:
        st.info(t("store_busy", lang), icon="⏳")
        return

    if history.empty:
        st.info(t("chart_no_history", lang))
        return

    row = signals[signals["ticker"] == choice].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("c_close", lang), f"{row['close']:,.0f}" if pd.notna(row["close"]) else "—")
    c2.metric(t("c_tier", lang), row["tier"] or "—")
    c3.metric(t("c_depth", lang), int(row["alignment_depth"]))
    c4.metric(t("c_rsi", lang), f"{row['rsi14']:.1f}" if pd.notna(row["rsi14"]) else "—")

    if row["data_quality"] == "insufficient_history":
        st.warning(t("warn_short", lang, bars=int(row["bars_available"])), icon="⚠️")
    elif row["data_quality"] == "suspect":
        st.error(t("warn_suspect", lang), icon="⚠️")
    elif row["data_quality"] == "stale":
        st.warning(t("warn_stale", lang), icon="⚠️")

    st.plotly_chart(
        build_combined_figure(history, choice, theme=theme, show_extras=show_extras),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    st.caption(t("chart_controls", lang))
    st.caption(t("chart_price_note", lang))
    st.caption(t("chart_fitted_note", lang))
    st.caption(t("chart_rsi_note", lang))

    with st.expander(t("raw_rows", lang)):
        st.dataframe(
            history.tail(20).iloc[::-1], use_container_width=True, hide_index=True
        )


# ---------------------------------------------------------------------------
# data health
# ---------------------------------------------------------------------------

def render_health(data: dict, lang: str) -> None:
    st.subheader(t("health_heading", lang))
    st.caption(t("health_caption", lang))

    quality = data["quality"]
    if quality.empty or quality["bars"].sum() == 0:
        st.info(t("health_none", lang))
    else:
        st.dataframe(quality, use_container_width=True, hide_index=True)

    counts = data["counts"]
    if not counts.empty:
        st.markdown(t("health_perday", lang))
        st.line_chart(
            counts.set_index("date")[
                ["depth5_gated", "depth4plus_gated", "depth3plus_gated"]
            ]
        )
        st.caption(t("health_perday_note", lang))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    lang = _pick_language()
    theme = _pick_theme(lang)

    st.title(t("app_title", lang))
    st.caption(t("app_caption", lang))

    try:
        data = _load()
    except StoreBusy:
        # The nightly sync is writing. One writer or many readers, never both.
        # A few minutes an evening, and the honest thing to say is what is
        # happening rather than to show a stack trace or, worse, stale figures
        # dressed as current ones.
        st.info(t("store_busy", lang), icon="⏳")
        if st.button(t("store_busy_retry", lang)):
            st.cache_data.clear()
            st.rerun()
        return

    if data is None:
        st.error(
            f"{t('no_store', lang, path=DB_PATH)}\n\n{t('no_store_help', lang)}\n\n"
            "```\n.\\idxcore.cmd init-db\n"
            ".\\idxcore.cmd sync-universe --from-file Daftar_Saham.xlsx\n"
            ".\\idxcore.cmd backfill --years 10\n"
            ".\\idxcore.cmd compute\n```"
        )
        return

    render_freshness(data["freshness"], data["universe"], lang)

    if data["signals"].empty:
        st.warning(t("no_signals", lang), icon="⚠️")
        return

    screener, guide, health = st.tabs(
        [t("tab_screener", lang), t("tab_guide", lang), t("tab_health", lang)]
    )
    with screener:
        matches = render_screener(data["signals"], lang, data["measured"])
        render_detail(data["signals"], matches, lang, theme)
    with guide:
        render_guide(lang)
    with health:
        render_health(data, lang)


if __name__ == "__main__":
    main()
