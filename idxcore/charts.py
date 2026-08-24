"""Chart construction, independent of any UI framework.

Kept out of the Streamlit app so the figures can be built, tested and exported
without a browser or a running server. The dashboard imports these; so does
`scripts/export_chart.py`.

These functions only *draw* what the store already holds. They compute nothing —
no indicator is derived here, in keeping with invariant 1.

On the dark background
----------------------
The MA5 line is white, which is invisible on a white page. Rather than quietly
substituting a different colour, the chart paints its own dark panel. That is
the convention for trading charts anyway, and it is what makes all five moving
averages legible at once.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

#: Deliberately far apart in hue so five lines stay tellable apart at a glance.
MA_COLOURS = {
    "ma5": "#FFFFFF",    # white
    "ma20": "#FF9800",   # orange
    "ma50": "#2196F3",   # blue
    "ma100": "#FF1744",  # red
    "ma200": "#B388FF",  # violet
}

MA_WIDTHS = {"ma5": 1.6, "ma20": 1.4, "ma50": 1.4, "ma100": 1.4, "ma200": 1.8}

#: Candles are muted so the MA lines read as the foreground. The down candle is
#: a desaturated brick rather than a pure red, so it cannot be confused with the
#: vivid red MA100 line.
UP_COLOUR = "#26A69A"
DOWN_COLOUR = "#C1554E"

PANEL_BG = "#131722"
GRID = "#2A2E39"
AXIS_TEXT = "#B2B5BE"

#: The 50 line is the gate criteria 4/5/6 hang on, so it gets a colour of its
#: own rather than the default black, which vanished against the dark panel.
GATE_COLOUR = "#FFD54F"
OVERBOUGHT_COLOUR = "#EF9A9A"
OVERSOLD_COLOUR = "#80CBC4"

#: RSI line colour, named once because the snapshot serialises it too.
RSI_COLOUR = "#7E9BFF"

#: Corporate-action badges. Fixed colours that read on both themes — a calm
#: blue for a dividend, orange for a split. There is no earnings badge because
#: earnings dates are not in the store, and inventing them is not on.
DIVIDEND_MARKER = "#1E88E5"
SPLIT_MARKER = "#F57C00"

#: The trend ribbon's four states, coloured as the owner's screenshots draw them
#: (see indicators.trend_ribbon). Same on both themes — the meaning is the same.
#: 3 strong-up, 2 weak-up, 1 weak-down, 0 strong-down.
RIBBON_COLOURS = {3: "#4CAF50", 2: "#2962FF", 1: "#F4A4A4", 0: "#FF5252"}
#: The capitulation histogram's dark-red squares.
CAPITULATION_COLOUR = "#8B0000"


#: Plotly config for every chart the app renders, matching what a trader
#: expects from a charting platform rather than from a plotting library.
#:
#: Plotly defaults to box-zoom on drag, which means the only way to move
#: through history is to zoom out and back in somewhere else. Dragging should
#: move the chart and the wheel should zoom it — so ``dragmode`` is ``pan``
#: (set in the layout) and ``scrollZoom`` is on here.
#:
#: Pass it wherever a figure is rendered: ``st.plotly_chart(fig,
#: config=PLOTLY_CONFIG)`` and ``fig.to_html(config=PLOTLY_CONFIG)``. It lives
#: beside the figure builders because it is part of how the chart behaves, not
#: a property of the page that happens to host it.
PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    # Double-click returns to the full stored range, which is the way back
    # after panning somewhere unrecognisable.
    "doubleClick": "reset",
    # Box and lasso select do nothing on a price chart; leaving them on the
    # modebar invites a click that appears to break the drag behaviour.
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
}


def _layout(p: dict, **overrides) -> dict:
    """Base layout for one theme's palette.

    Everything colour-bearing reads from `p` (see :func:`palette`) so the same
    figure builders serve both the dark default and the light theme.
    """
    layout = dict(
        paper_bgcolor=p["panel"],
        plot_bgcolor=p["panel"],
        font=dict(color=p["axis_text"]),
        xaxis=dict(gridcolor=p["grid"], linecolor=p["grid"], zeroline=False),
        yaxis=dict(gridcolor=p["grid"], linecolor=p["grid"], zeroline=False),
        hovermode="x unified",
        # Drag pans; the wheel zooms. Plotly's default is box-zoom on drag,
        # which is the wrong instinct for a time series you read by scrubbing
        # backwards through it.
        dragmode="pan",
    )
    layout.update(overrides)
    return layout


def non_trading_breaks(dates: pd.Series) -> list:
    """Calendar days with no bar, so the x-axis can skip them.

    Weekends *and* exchange holidays. Rather than assuming a Mon-Fri calendar,
    this takes the dates actually present and treats everything missing between
    the first and last as a break — so Idul Fitri closures collapse too, and no
    flat gap is drawn across a week the market was shut.
    """
    stamps = pd.to_datetime(pd.Series(dates)).dt.normalize()
    if stamps.empty:
        return []
    full = pd.date_range(stamps.min(), stamps.max(), freq="D")
    missing = full.difference(pd.DatetimeIndex(stamps.unique()))
    return list(missing)


def _apply_trading_calendar(fig: go.Figure, dates: pd.Series) -> None:
    breaks = non_trading_breaks(dates)
    if breaks:
        fig.update_xaxes(rangebreaks=[dict(values=breaks)])


def build_price_figure(
    history: pd.DataFrame,
    ticker: str,
    *,
    height: int = 480,
    title: Optional[str] = None,
    theme: str = "dark",
) -> go.Figure:
    """Candlesticks plus whichever moving averages actually exist.

    An MA with no values is omitted rather than drawn as a flat line at zero —
    a young listing has no MA200, and pretending otherwise is the same
    silent-downgrade bug in visual form.
    """
    p = palette(theme)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=history["date"],
            open=history["open"],
            high=history["high"],
            low=history["low"],
            close=history["close"],
            name=ticker,
            increasing_line_color=p["up"],
            decreasing_line_color=p["down"],
            increasing_fillcolor=p["up"],
            decreasing_fillcolor=p["down"],
        )
    )

    for column, colour in p["ma"].items():
        if column in history and history[column].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=history["date"],
                    y=history[column],
                    name=column.upper(),
                    mode="lines",
                    line=dict(width=p["ma_width"][column], color=colour),
                )
            )

    layout = _layout(
        p,
        height=height,
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    # Only set the key when there is a title. Passing title=None survives as a
    # literal "undefined" once Streamlit applies its own template over the top.
    if title:
        layout["title"] = title
    fig.update_layout(**layout)
    # Price axis on the right, like TradingView; automargin reserves the space
    # so the labels are not clipped by the l=0/r=0 margins.
    fig.update_yaxes(
        title_text="Price (IDR, unadjusted)", gridcolor=p["grid"],
        side="right", automargin=True,
    )
    _apply_trading_calendar(fig, history["date"])
    return fig


#: Ichimoku draws the two Senkou spans this many bars ahead of the bar they
#: were computed on. The store keeps them unshifted; the shift lives here.
KUMO_SHIFT = 26

#: Taken from the Pine indicator rather than picked by eye. TradingView's
#: color.blue is #2962FF, color.red #FF5252, color.gray #787B86, and the script
#: sets transparency 28 on the band lines, 82 on the held-level dots and 84 on
#: the Bollinger fill — Pine transparency is the inverse of alpha.
FCB_UP = "rgba(41, 98, 255, 0.72)"
#: The lower band is orange, not the script's red: a red line on top of red down
#: candles is unreadable. Blue upper stays as the script draws it.
FCB_DOWN = "rgba(251, 140, 0, 0.9)"       # #FB8C00
FCB_DOT_UP = "rgba(41, 98, 255, 0.18)"
FCB_DOT_DOWN = "rgba(251, 140, 0, 0.5)"   # orange held-low dots
BB_FILL = "rgba(120, 123, 134, 0.16)"

#: The script draws the bands and dots ``-(pattern + 1)`` bars back, because a
#: fractal is not confirmable until that many bars after it. The values are
#: stored at the date they were computed; the shift is applied here, on render.
#: pattern=1 (see indicators.FCB_PATTERN) -> -(1 + 1).
FCB_DRAW_OFFSET = -2


# ---------------------------------------------------------------------------
# themes
# ---------------------------------------------------------------------------
#
# One palette per theme, and one function that hands it out. This is the single
# source of truth for chart colour: the Streamlit builders below read it, and
# the shared snapshot serialises the very same dict (see build_snapshot.py), so
# the two surfaces cannot drift into different-coloured charts.
#
# The dark theme is assembled from the module constants above so nothing is
# stated twice; the light theme carries its own values. The only hard rule is
# that MA5 must not stay white — invisible on a white page — which is why the
# light theme's ma5 is near-black.

_LIGHT_PALETTE = {
    "panel": "#FFFFFF",
    "grid": "#E0E3EB",
    "axis_text": "#434651",
    "up": "#26A69A",
    "down": "#C1554E",       # kept distinct from the vivid MA100 red, as in dark
    "gate": "#F9A825",       # amber: #FFD54F yellow vanishes on white
    "overbought": "#E57373",
    "oversold": "#00897B",
    "bb_fill": "rgba(120, 123, 134, 0.16)",
    "fcb_up": FCB_UP,
    "fcb_down": FCB_DOWN,
    "fcb_dot_up": "rgba(41, 98, 255, 0.28)",    # the 0.18 dots disappear on white
    "fcb_dot_down": "rgba(251, 140, 0, 0.55)",  # orange held-low dots
    "volume_up": "#26A69A",
    "volume_down": "#C1554E",
    "rsi": "#3F51B5",        # indigo: the pale #7E9BFF washes out on white
    "ma": {
        "ma5": "#131722",    # near-black, not white — see the rule above
        "ma20": "#FB8C00",
        "ma50": "#1E88E5",
        "ma100": "#E53935",
        "ma200": "#7E57C2",
    },
    "ma_width": dict(MA_WIDTHS),
    "badge_close_text": "#FFFFFF",
    "badge_close_bg": "#434651",
    "badge_ma_text": "#FFFFFF",
    "dividend_marker": DIVIDEND_MARKER,
    "split_marker": SPLIT_MARKER,
    "ribbon": RIBBON_COLOURS,
    "capitulation": CAPITULATION_COLOUR,
}

THEMES = {
    "dark": {
        "panel": PANEL_BG,
        "grid": GRID,
        "axis_text": AXIS_TEXT,
        "up": UP_COLOUR,
        "down": DOWN_COLOUR,
        "gate": GATE_COLOUR,
        "overbought": OVERBOUGHT_COLOUR,
        "oversold": OVERSOLD_COLOUR,
        "bb_fill": BB_FILL,
        "fcb_up": FCB_UP,
        "fcb_down": FCB_DOWN,
        "fcb_dot_up": FCB_DOT_UP,
        "fcb_dot_down": FCB_DOT_DOWN,
        "volume_up": UP_COLOUR,
        "volume_down": DOWN_COLOUR,
        "rsi": RSI_COLOUR,
        "ma": dict(MA_COLOURS),
        "ma_width": dict(MA_WIDTHS),
        "badge_close_text": "#FFFFFF",
        "badge_close_bg": "#2A2E39",
        "badge_ma_text": "#0E1117",
        "dividend_marker": DIVIDEND_MARKER,
        "split_marker": SPLIT_MARKER,
        "ribbon": RIBBON_COLOURS,
        "capitulation": CAPITULATION_COLOUR,
    },
    "light": _LIGHT_PALETTE,
}


def palette(theme: str = "dark") -> dict:
    """The colour set for one theme. Raises on an unknown name rather than
    silently falling back, so a typo fails loudly instead of shipping the wrong
    theme."""
    try:
        return THEMES[theme]
    except KeyError as exc:
        raise ValueError(
            f"unknown theme {theme!r}; expected one of {sorted(THEMES)}"
        ) from exc


def shift_forward(dates: pd.Series, values: pd.Series, bars: int) -> tuple[list, list]:
    """Move a series `bars` trading days forward, extending past the last bar.

    The projection uses the spacing of real trading days rather than calendar
    days, so the cloud lands on sessions the exchange will actually hold rather
    than on weekends.
    """
    stamps = list(pd.to_datetime(pd.Series(dates)))
    if len(stamps) < 2:
        return stamps, list(values)

    # Median gap between stored bars, as a stand-in for the next few sessions.
    gaps = pd.Series(stamps).diff().dropna()
    step = gaps.median() if not gaps.empty else pd.Timedelta(days=1)
    future = [stamps[-1] + step * (i + 1) for i in range(bars)]
    return stamps + future, [None] * bars + list(values)


def _add_bollinger(fig, history: pd.DataFrame, p: dict) -> None:
    """The grey cloud: Bollinger(20, 2), band lines invisible, the fill shaded.

    The Pine script plots both edges at transparency 100 and shades between
    them, so what the reader sees is the fill alone. Drawn first, under
    everything, for the same reason.
    """
    if not {"bb_upper", "bb_lower"} <= set(history.columns):
        return
    if history["bb_upper"].isna().all():
        return

    fig.add_trace(
        go.Scatter(
            x=history["date"], y=history["bb_upper"], name="BB upper",
            mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=history["date"], y=history["bb_lower"], name="BB(20,2)",
            mode="lines", line=dict(width=0), fill="tonexty", fillcolor=p["bb_fill"],
            hoverinfo="skip", showlegend=True,
        ),
        row=1, col=1,
    )


def _add_fractal_bands(fig, history: pd.DataFrame, p: dict) -> None:
    """Fractal Chaos Bands, and the levels they leave behind.

    Both are shifted back by ``FCB_DRAW_OFFSET`` to sit on the fractal candle
    they describe, which is what the script does and what makes the line agree
    with the price it is drawn against.
    """
    dates = list(history["date"])

    def shifted(column):
        """Move the series left, so bar t shows the value computed at t+2."""
        values = list(history[column])
        drop = -FCB_DRAW_OFFSET
        return values[drop:] + [None] * drop

    for column, colour, label, dotted in (
        ("fcb_upper", p["fcb_up"], "FCB upper", False),
        ("fcb_lower", p["fcb_down"], "FCB lower", False),
        ("fcb_dot_upper", p["fcb_dot_up"], "Held high", True),
        ("fcb_dot_lower", p["fcb_dot_down"], "Held low", True),
    ):
        if column not in history or history[column].isna().all():
            continue
        # Solid bands: straight lines between points (TradingView plot.style_line),
        # so a level move is a single diagonal, not a stiff right-angle step.
        # Held levels: a continuous dotted horizontal line that runs across the
        # bars it held for — `hv` keeps each level flat and steps at a change,
        # matching the dotted projections in the reference charts.
        line = (
            dict(width=1.2, color=colour, dash="dot", shape="hv")
            if dotted
            else dict(width=1.4, color=colour, shape="linear")
        )
        fig.add_trace(
            go.Scatter(
                x=dates, y=shifted(column), name=label,
                mode="lines", line=line,
                connectgaps=False,
                hoverinfo="skip" if dotted else None,
                showlegend=not dotted,
            ),
            row=1, col=1,
        )


def _add_corporate_actions(fig, history: pd.DataFrame) -> None:
    """Dividend (D) and split (S) badges under the bar they fell on.

    Drawn straight from the stored raw columns — ``dividend`` (IDR per share)
    and ``split_factor`` (a multiplier, 1.0 meaning none). Nothing is computed
    here, in keeping with invariant 1. Earnings dates are not stored, so there
    is deliberately no E badge: a made-up date is worse than a missing one.
    """
    if history.empty or "low" not in history:
        return
    low = pd.to_numeric(history["low"], errors="coerce")
    date = history["date"]

    def badge(mask: pd.Series, label: str, colour: str, hover: list) -> None:
        if not mask.any():
            return
        # Just under the low, proportional so the badge holds at any price scale.
        fig.add_trace(
            go.Scatter(
                x=date[mask], y=low[mask] * 0.985,
                mode="markers+text", text=[label] * int(mask.sum()),
                textposition="bottom center", textfont=dict(size=9, color=colour),
                marker=dict(size=6, color=colour, symbol="circle"),
                name="Dividend" if label == "D" else "Split",
                hovertext=hover, hoverinfo="text",
            ),
            row=1, col=1,
        )

    if "dividend" in history:
        div = pd.to_numeric(history["dividend"], errors="coerce").fillna(0.0)
        mask = div > 0
        badge(mask, "D", DIVIDEND_MARKER,
              [f"Dividend {v:,.2f}" for v in div[mask]])
    if "split_factor" in history:
        sf = pd.to_numeric(history["split_factor"], errors="coerce").fillna(1.0)
        mask = sf.ne(1.0)
        badge(mask, "S", SPLIT_MARKER,
              [f"Split ×{v:g}" for v in sf[mask]])


def _add_price_badges(fig, history: pd.DataFrame, p: dict) -> None:
    """Right-edge labels for the last close and each moving average."""
    if history.empty:
        return
    last = history.iloc[-1]

    levels = [("close", last.get("close"), p["badge_close_text"], p["badge_close_bg"])]
    for column, colour in p["ma"].items():
        levels.append((column.upper(), last.get(column), p["badge_ma_text"], colour))

    for label, value, text_colour, bg in levels:
        if value is None or pd.isna(value):
            continue
        fig.add_annotation(
            # "x domain", not "paper". Passing xref="paper" together with
            # row/col makes plotly rewrite the ref to the subplot's own axis,
            # and x=1.0 is then read as a *date* one millisecond after the Unix
            # epoch. The axis stretched back to 1970 and every chart collapsed
            # into a sliver at the right-hand edge.
            x=1.0, xref="x domain",
            y=float(value), yref="y",
            text=f" {float(value):,.0f} ",
            showarrow=False, xanchor="left",
            font=dict(size=10, color=text_colour),
            bgcolor=bg, borderpad=2,
            row=1, col=1,
        )


def _add_ribbon(fig, history: pd.DataFrame, p: dict, row: int) -> None:
    """The trend ribbon: one unit-height cell per bar, coloured by state.

    A fitted approximation of the owner's screenshot ribbon (see
    indicators.trend_ribbon). With the figure's bargap at 0 the cells touch, so
    the states read as continuous blocks rather than a picket fence. Warm-up
    bars (no state) are drawn transparent.
    """
    if "ribbon" not in history or history["ribbon"].isna().all():
        return
    colours = [
        p["ribbon"].get(int(s), "rgba(0,0,0,0)") if pd.notna(s) else "rgba(0,0,0,0)"
        for s in history["ribbon"]
    ]
    fig.add_trace(
        go.Bar(
            x=history["date"], y=[1] * len(history),
            marker_color=colours, marker_line_width=0,
            name="Trend", showlegend=False, hoverinfo="skip",
        ),
        row=row, col=1,
    )


def _add_capitulation(fig, history: pd.DataFrame, p: dict, row: int) -> None:
    """The capitulation histogram: a fixed -1 bar on each flagged day.

    Placeholder trigger (see indicators.capitulation_marker). The fixed height
    matches the screenshots, where every square sits at the same depth.
    """
    if "capitulation" not in history or history["capitulation"].isna().all():
        return
    flag = pd.to_numeric(history["capitulation"], errors="coerce").fillna(0)
    fig.add_trace(
        go.Bar(
            x=history["date"], y=[-1 if v else 0 for v in flag],
            marker_color=p["capitulation"], marker_line_width=0,
            name="Capitulation", showlegend=False, hoverinfo="skip",
        ),
        row=row, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=p["axis_text"],
                  line_width=1, row=row, col=1)


def build_combined_figure(
    history: pd.DataFrame,
    ticker: str,
    *,
    height: int = 820,
    theme: str = "dark",
    show_extras: bool = False,
) -> go.Figure:
    """Price and volume stacked on one shared time axis; MAs and RSI optional.

    ``show_extras=False`` (the default) draws the chart the owner reads from —
    candles, the grey Bollinger cloud, the fractal bands, volume, and the two
    lower panels from his TradingView setup: the fitted trend ribbon and the
    capitulation histogram — with no moving averages and no RSI panel. With it
    on the five MAs return to the price panel and an RSI panel slots in above
    the ribbon, for reading the six criteria off the chart. The panels share one
    time axis so the crosshair lines up and a volume spike reads against its bar.
    """
    from plotly.subplots import make_subplots

    p = palette(theme)

    # Panels, top to bottom: price, volume, [RSI when extras], trend ribbon,
    # capitulation histogram. The ribbon and histogram are always drawn — they
    # are what the owner's TradingView screenshots show below the price.
    rsi_row = 3 if show_extras else None
    ribbon_row = 4 if show_extras else 3
    hist_row = 5 if show_extras else 4
    rows = 5 if show_extras else 4
    row_heights = (
        [0.48, 0.13, 0.15, 0.09, 0.15] if show_extras
        else [0.58, 0.16, 0.10, 0.16]
    )
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=row_heights,
    )

    # Bollinger and the fractal bands replace the Kumo and Donchian channel in
    # the price panel. Both of those are still computed and stored - nothing was
    # thrown away - they are simply not what this chart draws any more.
    _add_bollinger(fig, history, p)
    _add_fractal_bands(fig, history, p)

    fig.add_trace(
        go.Candlestick(
            x=history["date"],
            open=history["open"], high=history["high"],
            low=history["low"], close=history["close"],
            name=ticker,
            increasing_line_color=p["up"], decreasing_line_color=p["down"],
            increasing_fillcolor=p["up"], decreasing_fillcolor=p["down"],
        ),
        row=1, col=1,
    )
    if show_extras:
        for column, colour in p["ma"].items():
            if column in history and history[column].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=history["date"], y=history[column], name=column.upper(),
                        mode="lines",
                        line=dict(width=p["ma_width"][column], color=colour),
                    ),
                    row=1, col=1,
                )

    _add_corporate_actions(fig, history)

    # Volume, coloured by whether the bar closed up or down — the same
    # green/red the candles use, so the two panels read as one picture.
    if "volume" in history:
        closes = pd.to_numeric(history["close"], errors="coerce")
        rising = closes >= closes.shift(1).fillna(closes.iloc[0])
        fig.add_trace(
            go.Bar(
                x=history["date"],
                y=pd.to_numeric(history["volume"], errors="coerce"),
                name="Volume",
                marker_color=[p["volume_up"] if up else p["volume_down"] for up in rising],
                marker_line_width=0,
                opacity=0.75,
                showlegend=False,
            ),
            row=2, col=1,
        )

    if show_extras and "rsi14" in history and history["rsi14"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=history["date"], y=history["rsi14"], name="RSI(14)",
                mode="lines", line=dict(color=p["rsi"], width=1.6),
                showlegend=False,
            ),
            row=rsi_row, col=1,
        )
        for level, colour, dash in (
            (50, p["gate"], "dash"),
            (70, p["overbought"], "dot"),
            (30, p["oversold"], "dot"),
        ):
            fig.add_hline(
                y=level, line_dash=dash, line_color=colour,
                line_width=1.6 if level == 50 else 1, row=rsi_row, col=1,
            )

    _add_ribbon(fig, history, p, ribbon_row)
    _add_capitulation(fig, history, p, hist_row)

    fig.update_layout(
        **_layout(
            p,
            height=height,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.02, yanchor="bottom"),
            # 0 so the ribbon cells touch and read as continuous blocks; the
            # volume bars touching too is normal for a trading chart.
            bargap=0.0,
        )
    )
    _add_price_badges(fig, history, p)
    fig.update_xaxes(rangeslider_visible=False)
    # Axes on the right, like TradingView. automargin keeps the labels off the
    # plot despite the l=0/r=0 margins.
    fig.update_yaxes(
        title_text="Price (IDR)", gridcolor=p["grid"],
        side="right", automargin=True, row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Vol", gridcolor=p["grid"],
        side="right", automargin=True, row=2, col=1,
    )
    if show_extras:
        fig.update_yaxes(
            title_text="RSI", gridcolor=p["grid"], range=[0, 100],
            side="right", automargin=True, row=rsi_row, col=1,
        )
    # The ribbon is a colour strip: no scale to read, so its y-axis is hidden.
    fig.update_yaxes(
        title_text="Trend", showticklabels=False, range=[0, 1],
        side="right", automargin=True, row=ribbon_row, col=1,
    )
    fig.update_yaxes(
        title_text="Cap.", gridcolor=p["grid"], range=[-3, 0.2],
        side="right", automargin=True, row=hist_row, col=1,
    )

    # Nothing is drawn past the last bar any more, so the axis stops there.
    _apply_trading_calendar(fig, history["date"])
    return fig


def build_rsi_figure(
    history: pd.DataFrame, *, height: int = 220, theme: str = "dark"
) -> Optional[go.Figure]:
    """RSI(14) with the gate marked. Returns None when RSI is undefined throughout."""
    if "rsi14" not in history or history["rsi14"].isna().all():
        return None

    p = palette(theme)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["rsi14"],
            name="RSI(14)",
            mode="lines",
            line=dict(color=p["rsi"], width=1.6),
        )
    )
    # 50 is the gate criteria 4/5/6 hang on, so it is drawn more prominently
    # than the conventional 30/70 bands, which this system does not use.
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color=p["gate"],
        line_width=1.6,
        annotation_text="gate 50",
        annotation_font_color=p["gate"],
    )
    fig.add_hline(y=70, line_dash="dot", line_color=p["overbought"], line_width=1)
    fig.add_hline(y=30, line_dash="dot", line_color=p["oversold"], line_width=1)

    fig.update_layout(
        **_layout(
            p,
            height=height,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_range=[0, 100],
            showlegend=False,
        )
    )
    fig.update_yaxes(title_text="RSI(14)", gridcolor=p["grid"], side="right", automargin=True)
    _apply_trading_calendar(fig, history["date"])
    return fig
