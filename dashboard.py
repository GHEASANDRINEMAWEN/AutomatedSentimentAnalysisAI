"""Africa Insights — Tourism Perception dashboard (Streamlit + Plotly).

Reads data/records.csv (produced by run.py) and presents a filterable, premium
analytics view of tourism sentiment per country, aspect, source, and time.

Run:
    python -m streamlit run dashboard.py

The data-shaping helpers (load_data / filter / aggregate) are plain functions
with no Streamlit calls, so they can be imported and tested independently.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_FILE = Path(__file__).parent / "data" / "records.csv"

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
TEAL = "#127B82"        # brand accent
GREEN = "#2E9E5B"       # positive
GREY = "#9AA7A8"        # neutral
RED = "#D1495B"         # negative
VIOLET = "#7C6FD6"      # volume / engagement
INK = "#1C2B2E"         # headings
MUTED = "#5E7174"       # labels / captions
BORDER = "#E6EBEB"
FRAME = "#F5F7F8"
CARD = "#FFFFFF"

SENTIMENTS = ["positive", "neutral", "negative"]
SENTIMENT_COLORS = {"positive": GREEN, "neutral": GREY, "negative": RED}

# Qualitative palette for per-country series in the regional-comparison view.
COUNTRY_PALETTE = [
    "#127B82", "#7C6FD6", "#E8A33D", "#2E9E5B",
    "#D1495B", "#3D7DCA", "#C45BAA", "#1C8C8C",
]

ALL_ASPECTS = [
    "food", "scenery", "safety", "wildlife", "hospitality", "transport", "cost",
    "weather", "electricity", "water", "housing",
]
SOURCE_LABELS = {
    "youtube": "YouTube comments",
    "youtube_transcript": "YouTube transcripts",
    "google_hotels": "Google Hotels reviews",
    "tripadvisor": "Tripadvisor reviews",
    "reddit": "Reddit posts",
}

# Review sources (SerpApi) — these carry a 1-5 star rating.
REVIEW_SOURCES = ("google_hotels", "tripadvisor")

# Higher-level provider grouping for the sidebar "Data source" filter, so users
# can scope to a whole provider (all its sources at once) or mix them freely.
PROVIDER_GROUPS = {
    "YouTube": ["youtube", "youtube_transcript"],
    "Reviews (SerpApi)": ["google_hotels", "tripadvisor"],
    "Reddit": ["reddit"],
}


def provider_of(source: str) -> str:
    """Map a raw source to its provider group (falls back to the source name)."""
    for group, group_sources in PROVIDER_GROUPS.items():
        if source in group_sources:
            return group
    return source


YEAR_MIN, YEAR_MAX = 2015, 2026


# --------------------------------------------------------------------------- #
# Data shaping (no Streamlit UI — importable + testable)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data(path: str = str(DATA_FILE)) -> pd.DataFrame:
    """Load records.csv and derive the columns the dashboard needs."""
    df = pd.read_csv(path, encoding="utf-8-sig")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.strftime("%Y-%m")  # "YYYY-MM"
    df["date"] = df["timestamp"].dt.date

    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)
    df["engagement"] = pd.to_numeric(df["engagement"], errors="coerce").fillna(0).astype(int)
    df["relevance_kept"] = df["relevance_kept"].astype(str).str.lower().eq("true")
    df["aspects"] = df["aspects"].fillna("").astype(str)
    df["emotion"] = df["emotion"].fillna("").astype(str)
    df["sentiment_label"] = df["sentiment_label"].fillna("neutral").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Provider group (for the sidebar filter) + numeric star rating (reviews).
    df["provider"] = df["source"].map(provider_of)
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    else:
        df["rating"] = pd.Series([pd.NA] * len(df), dtype="Float64")
    return df


def filter_data(df, *, country, year_range, aspects, sentiments, sources, kept_only):
    """Apply the sidebar filters and return the filtered frame."""
    out = df
    if country and country != "All":
        out = out[out["country"] == country]
    if kept_only:
        out = out[out["relevance_kept"]]
    lo, hi = year_range
    out = out[(out["year"] >= lo) & (out["year"] <= hi)]
    if sentiments:
        out = out[out["sentiment_label"].isin(sentiments)]
    if sources:
        out = out[out["source"].isin(sources)]
    if aspects:
        pattern = "|".join(rf"(?:^|,){a}(?:,|$)" for a in aspects)
        out = out[out["aspects"].str.contains(pattern, regex=True, na=False)]
    return out


def sentiment_split(df) -> dict:
    counts = df["sentiment_label"].value_counts().to_dict()
    return {s: int(counts.get(s, 0)) for s in SENTIMENTS}


def summarize(df) -> dict:
    """Headline metrics for a frame."""
    n = len(df)
    split = sentiment_split(df)
    pos = 100 * split["positive"] / n if n else 0.0
    neg = 100 * split["negative"] / n if n else 0.0
    neu = 100 * split["neutral"] / n if n else 0.0
    net = pos - neg
    perception = round((df["sentiment_score"].mean() + 1) / 2 * 100) if n else 0
    return dict(n=n, pos=pos, neg=neg, neu=neu, net=net,
                perception=perception, split=split)


def aspect_sentiment_pct(df) -> pd.DataFrame:
    """Long frame of aspect x sentiment_label with within-aspect percentages."""
    exploded = df.assign(aspect=df["aspects"].str.split(",")).explode("aspect")
    exploded = exploded[exploded["aspect"].isin(ALL_ASPECTS)]
    if exploded.empty:
        return pd.DataFrame(columns=["aspect", "sentiment_label", "count", "pct", "total"])
    grouped = exploded.groupby(["aspect", "sentiment_label"]).size().reset_index(name="count")
    totals = grouped.groupby("aspect")["count"].transform("sum")
    grouped["total"] = totals
    grouped["pct"] = 100 * grouped["count"] / totals
    return grouped


def aspect_net(df) -> pd.DataFrame:
    """Per-aspect net sentiment (positive% - negative%)."""
    g = aspect_sentiment_pct(df)
    if g.empty:
        return pd.DataFrame(columns=["aspect", "net", "positive", "negative", "total"])
    piv = g.pivot_table(index="aspect", columns="sentiment_label",
                        values="pct", fill_value=0.0)
    for s in SENTIMENTS:
        if s not in piv:
            piv[s] = 0.0
    piv["net"] = piv["positive"] - piv["negative"]
    piv["total"] = g.groupby("aspect")["total"].first()
    return piv.reset_index().sort_values("net")


# --------------------------------------------------------------------------- #
# Reviews / star-rating helpers (SerpApi)
# --------------------------------------------------------------------------- #
def review_frame(df) -> pd.DataFrame:
    """Rows from review sources that carry a usable 1-5 star rating."""
    if "rating" not in df.columns or df.empty:
        return df.iloc[0:0]
    return df[df["source"].isin(REVIEW_SOURCES) & df["rating"].notna()]


def expected_sentiment_from_rating(rating):
    """Sentiment a star rating implies: 4-5 positive, 3 neutral, 1-2 negative.

    Uses 3.5 / 2.5 cutoffs so fractional aggregate ratings bucket sensibly.
    """
    if pd.isna(rating):
        return None
    if rating >= 3.5:
        return "positive"
    if rating >= 2.5:
        return "neutral"
    return "negative"


def rating_vs_sentiment(df):
    """Agreement between the model's sentiment and the star-implied sentiment.

    Returns dict(agree, total, rate, confusion, frame) over rated reviews in
    view, or None when there are none.
    """
    r = review_frame(df).copy()
    r = r[r["sentiment_label"].isin(SENTIMENTS)]
    if r.empty:
        return None
    r["expected"] = r["rating"].map(expected_sentiment_from_rating)
    agree = int((r["expected"] == r["sentiment_label"]).sum())
    total = int(len(r))
    confusion = (pd.crosstab(r["expected"], r["sentiment_label"])
                 .reindex(index=SENTIMENTS, columns=SENTIMENTS, fill_value=0))
    return dict(agree=agree, total=total,
                rate=(100 * agree / total if total else 0.0),
                confusion=confusion, frame=r)


def time_series(df, granularity: str) -> pd.DataFrame:
    """Average sentiment + volume per period (granularity = 'Year' or 'Month')."""
    key = "year" if granularity == "Year" else "month"
    if df.empty:
        return pd.DataFrame(columns=["period", "avg_sentiment", "perception", "volume"])
    grouped = (
        df.groupby(key)
        .agg(avg_sentiment=("sentiment_score", "mean"), volume=("source", "size"))
        .reset_index()
        .rename(columns={key: "period"})
    )
    grouped["perception"] = (grouped["avg_sentiment"] + 1) / 2 * 100
    grouped["period"] = grouped["period"].astype(str)
    return grouped.sort_values("period")


# --------------------------------------------------------------------------- #
# Regional comparison (across countries)
# --------------------------------------------------------------------------- #
def country_color_map(countries) -> dict:
    return {c: COUNTRY_PALETTE[i % len(COUNTRY_PALETTE)]
            for i, c in enumerate(sorted(countries))}


def compare_metrics(df_all) -> pd.DataFrame:
    """One row per country with headline metrics, ranked by perception."""
    rows = []
    for country, g in df_all.groupby("country"):
        m = summarize(g)
        net = aspect_net(g)
        loved = net.loc[net["net"].idxmax(), "aspect"] if not net.empty else "—"
        crit = net.loc[net["net"].idxmin(), "aspect"] if not net.empty else "—"
        rows.append(dict(
            country=country, mentions=m["n"], perception=m["perception"],
            net=m["net"], positive=m["pos"], neutral=m["neu"], negative=m["neg"],
            top_loved=loved, top_critical=crit,
        ))
    if not rows:
        return pd.DataFrame(columns=[
            "country", "mentions", "perception", "net", "positive",
            "neutral", "negative", "top_loved", "top_critical"])
    return pd.DataFrame(rows).sort_values("perception", ascending=False)


def compare_aspect_net(df_all) -> pd.DataFrame:
    """Matrix (rows=country, cols=aspect) of net sentiment per aspect."""
    series = {}
    for country, g in df_all.groupby("country"):
        net = aspect_net(g)
        if not net.empty:
            series[country] = net.set_index("aspect")["net"]
    if not series:
        return pd.DataFrame(columns=ALL_ASPECTS)
    return pd.DataFrame(series).T.reindex(columns=ALL_ASPECTS)


# --------------------------------------------------------------------------- #
# Styling helpers
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"], button, input, textarea {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        [data-testid="stAppViewContainer"] { background: #F5F7F8; }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }

        [data-testid="stSidebar"] {
            background: #FFFFFF; border-right: 1px solid #E6EBEB;
        }
        [data-testid="stSidebar"] h2 {
            color: #127B82; font-size: 15px; text-transform: uppercase;
            letter-spacing: .06em; font-weight: 700;
        }

        /* Cards = bordered containers */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF;
            border: 1px solid #E6EBEB !important;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(16,40,40,.06), 0 1px 2px rgba(16,40,40,.04);
            padding: 18px 20px;
        }

        /* Header */
        .app-header {
            display: flex; justify-content: space-between; align-items: center;
            gap: 24px; margin-bottom: 22px;
        }
        .app-title { color: #127B82; font-size: 30px; font-weight: 700; line-height: 1.1; }
        .app-sub   { color: #5E7174; font-size: 14px; margin-top: 6px; }
        .score-box {
            background: #FFFFFF; border: 1px solid #E6EBEB; border-radius: 12px;
            padding: 12px 22px; text-align: right; min-width: 180px;
            box-shadow: 0 1px 3px rgba(16,40,40,.06);
        }
        .score-label { color: #5E7174; font-size: 11px; font-weight: 600;
            text-transform: uppercase; letter-spacing: .08em; }
        .score-row { display: flex; align-items: baseline; justify-content: flex-end; gap: 10px; }
        .score-value { color: #127B82; font-size: 40px; font-weight: 700; line-height: 1; }
        .score-delta { font-size: 14px; font-weight: 600; }

        /* Metric cards */
        .metric-card {
            background: #FFFFFF; border: 1px solid #E6EBEB; border-radius: 12px;
            padding: 18px 20px; box-shadow: 0 1px 3px rgba(16,40,40,.06);
            height: 100%;
        }
        .metric-label { color: #5E7174; font-size: 11px; font-weight: 600;
            text-transform: uppercase; letter-spacing: .08em; }
        .metric-value { color: #1C2B2E; font-size: 30px; font-weight: 700;
            margin-top: 6px; line-height: 1.1; }
        .metric-delta { font-size: 13px; font-weight: 600; margin-top: 6px; }
        .up   { color: #2E9E5B; }
        .down { color: #D1495B; }
        .flat { color: #9AA7A8; }

        /* Signals + comment cards */
        .signal-list { list-style: none; padding: 0; margin: 0; }
        .signal-list li { color: #1C2B2E; font-size: 14px; padding: 7px 0;
            border-bottom: 1px solid #F0F3F3; display: flex; gap: 10px; }
        .signal-list li:last-child { border-bottom: none; }
        .dot { width: 9px; height: 9px; border-radius: 50%; margin-top: 5px; flex: 0 0 auto; }

        .voice-card {
            border: 1px solid #E6EBEB; border-radius: 10px; padding: 14px 16px;
            margin-bottom: 12px; background: #FCFDFD;
        }
        .voice-meta { color: #5E7174; font-size: 12px; display: flex;
            justify-content: space-between; margin-bottom: 7px; }
        .voice-text { color: #1C2B2E; font-size: 14px; line-height: 1.45; }
        .voice-card a { color: #127B82; font-size: 12px; font-weight: 600;
            text-decoration: none; }
        .pill { font-size: 11px; font-weight: 700; padding: 2px 9px; border-radius: 20px;
            text-transform: uppercase; letter-spacing: .04em; }

        /* Expandable long comments in the aspect explorer */
        .voice-card details { margin: 0; }
        .voice-card details summary {
            cursor: pointer; color: #1C2B2E; font-size: 14px; line-height: 1.45;
            list-style-position: outside; }
        .voice-card details summary::-webkit-details-marker { color: #127B82; }
        .voice-card details summary:hover { color: #127B82; }
        .voice-card .more-link { color: #127B82; font-weight: 600; font-size: 12px; }

        .sec-title { color: #1C2B2E; font-size: 16px; font-weight: 700; margin: 0 0 2px 0; }
        .sec-cap   { color: #5E7174; font-size: 12.5px; margin: 0 0 12px 0; }

        [data-baseweb="tab-list"] { gap: 6px; }
        button[data-baseweb="tab"] { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=MUTED, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickcolor=BORDER,
                     title_font=dict(size=12, color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor="#EEF2F2", zeroline=False,
                     linecolor="rgba(0,0,0,0)", title_font=dict(size=12, color=MUTED))
    return fig


def delta_html(value, good_when="up", suffix="", fmt="{:+.1f}"):
    """Return a styled 'vs previous period' span. value is None -> neutral dash."""
    if value is None:
        return '<div class="metric-delta flat">— vs prev period</div>'
    if abs(value) < 0.05:
        cls, arrow = "flat", "→"
    elif (value > 0) == (good_when == "up"):
        cls, arrow = "up", ("▲" if value > 0 else "▼")
    else:
        cls, arrow = "down", ("▲" if value > 0 else "▼")
    return (f'<div class="metric-delta {cls}">{arrow} {fmt.format(value)}{suffix} '
            f'vs prev period</div>')


# --------------------------------------------------------------------------- #
# Render sections
# --------------------------------------------------------------------------- #
def render_header(df, cur, prev):
    if df.empty:
        sub = "South Africa · no records for current filters"
    else:
        sub = (f"South Africa · {df['date'].min()} → {df['date'].max()} "
               f"· {cur['n']:,} mentions")
    d = None if prev["n"] == 0 else cur["perception"] - prev["perception"]
    if d is None:
        score_delta = '<span class="score-delta flat">—</span>'
    else:
        cls = "up" if d > 0 else ("down" if d < 0 else "flat")
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "→")
        score_delta = f'<span class="score-delta {cls}">{arrow} {d:+d}</span>'
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <div class="app-title">Africa Insights — Tourism Perception</div>
            <div class="app-sub">{sub}</div>
          </div>
          <div class="score-box">
            <div class="score-label">Perception Score</div>
            <div class="score-row">
              <span class="score-value">{cur['perception']}</span>{score_delta}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(cur, prev):
    have_prev = prev["n"] > 0
    cards = [
        ("Net Sentiment", f"{cur['net']:+.0f}%",
         (cur["net"] - prev["net"]) if have_prev else None, "up", "pts"),
        ("Positive", f"{cur['pos']:.0f}%",
         (cur["pos"] - prev["pos"]) if have_prev else None, "up", "pts"),
        ("Negative", f"{cur['neg']:.0f}%",
         (cur["neg"] - prev["neg"]) if have_prev else None, "down", "pts"),
        ("Total Mentions", f"{cur['n']:,}",
         (cur["n"] - prev["n"]) if have_prev else None, "up", ""),
    ]
    cols = st.columns(4, gap="medium")
    for col, (label, value, d, good, suffix) in zip(cols, cards):
        if suffix == "" and d is not None:
            delta = delta_html(d, good, "", "{:+,.0f}")
        else:
            delta = delta_html(d, good, " " + suffix if suffix else "")
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>{delta}</div>',
            unsafe_allow_html=True,
        )


def render_ribbon(df, cur):
    st.markdown('<div class="sec-title">Sentiment mix</div>'
                '<div class="sec-cap">Share of positive / neutral / negative across all '
                'mentions in view.</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("No records for the current filters.")
        return
    total = cur["n"] or 1
    fig = go.Figure()
    for s in SENTIMENTS:
        count = cur["split"][s]
        fig.add_bar(
            x=[100 * count / total], y=["mix"], orientation="h",
            name=f"{s.capitalize()} · {count:,}", marker_color=SENTIMENT_COLORS[s],
            marker_cornerradius=6,
            hovertemplate=s + ": %{x:.1f}%%<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    style_fig(fig, height=130)
    fig.update_xaxes(visible=False, range=[0, 100])
    fig.update_yaxes(visible=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _gradient_line(fig, x, y, color, name, hover, secondary=False, rgb="18,123,130"):
    fig.add_trace(
        go.Scatter(
            x=x, y=y, name=name, mode="lines",
            line=dict(color=color, width=3, shape="spline", smoothing=0.6),
            fill="tozeroy",
            fillgradient=dict(
                type="vertical",
                colorscale=[[0.0, f"rgba({rgb},0.02)"], [1.0, f"rgba({rgb},0.32)"]],
            ),
            hovertemplate=hover,
        ),
        secondary_y=secondary,
    )


def render_trend(df, granularity):
    st.markdown(f'<div class="sec-title">Perception over time</div>'
                f'<div class="sec-cap">Perception score (0–100, teal) and mention volume '
                f'(violet), by {granularity.lower()}.</div>', unsafe_allow_html=True)
    series = time_series(df, granularity)
    if series.empty:
        st.info("No records for the current filters.")
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    _gradient_line(fig, series["period"], series["volume"], VIOLET, "Volume",
                   "%{x}<br>volume: %{y}<extra></extra>", secondary=False, rgb="124,111,214")
    _gradient_line(fig, series["period"], series["perception"], TEAL, "Perception",
                   "%{x}<br>perception: %{y:.0f}<extra></extra>", secondary=True,
                   rgb="18,123,130")
    fig.update_traces(marker=dict(size=6))
    style_fig(fig, height=340)
    fig.update_yaxes(title_text="Volume", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="Perception", range=[0, 100], secondary_y=True,
                     showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if granularity == "Month":
        st.caption("Monthly buckets can be sparse — read perception together with volume.")


def render_source_comparison(df):
    st.markdown('<div class="sec-title">Comments vs transcripts</div>'
                '<div class="sec-cap">Sentiment composition by signal source.</div>',
                unsafe_allow_html=True)
    if df.empty:
        st.info("No records for the current filters.")
        return
    grouped = df.groupby(["source", "sentiment_label"]).size().reset_index(name="count")
    grouped["label"] = grouped["source"].map(lambda s: SOURCE_LABELS.get(s, s))
    fig = go.Figure()
    for s in SENTIMENTS:
        sub = grouped[grouped["sentiment_label"] == s]
        fig.add_bar(x=sub["label"], y=sub["count"], name=s.capitalize(),
                    marker_color=SENTIMENT_COLORS[s], marker_cornerradius=6,
                    hovertemplate="%{x}<br>" + s + ": %{y}<extra></extra>")
    fig.update_layout(barmode="stack")
    style_fig(fig, height=340)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_aspect_stack(df):
    st.markdown('<div class="sec-title">Sentiment by aspect</div>'
                '<div class="sec-cap">Share of positive / neutral / negative within each '
                'travel topic.</div>', unsafe_allow_html=True)
    data = aspect_sentiment_pct(df)
    if data.empty:
        st.info("No aspect-tagged records for the current filters.")
        return
    order = (data.groupby("aspect")["count"].sum()
             .sort_values(ascending=True).index.tolist())
    fig = go.Figure()
    for s in SENTIMENTS:
        sub = data[data["sentiment_label"] == s].set_index("aspect").reindex(order)
        fig.add_bar(
            y=order, x=sub["pct"], name=s.capitalize(), orientation="h",
            marker_color=SENTIMENT_COLORS[s], marker_cornerradius=4,
            customdata=sub[["count", "total"]].fillna(0).values,
            hovertemplate="%{y} — " + s + ": %{customdata[0]} of %{customdata[1]} "
                          "(%{x:.0f}%%)<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    style_fig(fig, height=380)
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_aspect_net(df):
    st.markdown('<div class="sec-title">Net sentiment by aspect</div>'
                '<div class="sec-cap">Positive minus negative share — green leans loved, '
                'red leans criticised.</div>', unsafe_allow_html=True)
    net = aspect_net(df)
    if net.empty:
        st.info("No aspect-tagged records for the current filters.")
        return
    net = net.sort_values("net")
    colors = [GREEN if v >= 0 else RED for v in net["net"]]
    fig = go.Figure(go.Bar(
        y=net["aspect"], x=net["net"], orientation="h",
        marker_color=colors, marker_cornerradius=6,
        text=[f"{v:+.0f}%" for v in net["net"]], textposition="outside",
        textfont=dict(color=INK, size=12),
        customdata=net["total"],
        hovertemplate="%{y}<br>net: %{x:+.0f}%%<br>%{customdata} mentions<extra></extra>",
    ))
    style_fig(fig, height=380)
    fig.update_xaxes(ticksuffix="%", zeroline=True, zerolinecolor=BORDER, zerolinewidth=1)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def build_signals(df, cur, prev):
    """Auto-generated takeaway bullets: (color, text)."""
    out = []
    net = aspect_net(df)
    if not net.empty and net["total"].sum() > 0:
        best = net.loc[net["net"].idxmax()]
        worst = net.loc[net["net"].idxmin()]
        most = net.loc[net["total"].idxmax()]
        out.append((GREEN, f"<b>{best['aspect'].capitalize()}</b> is the most loved aspect "
                           f"({best['net']:+.0f}% net, {int(best['total'])} mentions)."))
        out.append((RED, f"<b>{worst['aspect'].capitalize()}</b> draws the most criticism "
                         f"({worst['net']:+.0f}% net, {int(worst['total'])} mentions)."))
        out.append((TEAL, f"<b>{most['aspect'].capitalize()}</b> is the most discussed topic "
                          f"with {int(most['total'])} mentions."))
    if prev["n"] > 0:
        d = cur["perception"] - prev["perception"]
        if d > 0:
            out.append((GREEN, f"Perception is <b>up {d:+d} points</b> vs the previous period."))
        elif d < 0:
            out.append((RED, f"Perception is <b>down {d:+d} points</b> vs the previous period."))
        else:
            out.append((GREY, "Perception is flat vs the previous period."))
    emo = df.loc[df["emotion"].ne("neutral") & df["emotion"].ne(""), "emotion"]
    if not emo.empty:
        top = emo.value_counts().idxmax()
        out.append((VIOLET, f"Beyond neutral, <b>{top}</b> is the dominant emotion in the "
                           f"conversation."))
    out.append((GREEN if cur["net"] >= 0 else RED,
                f"Overall, <b>{cur['pos']:.0f}%</b> of mentions are positive vs "
                f"<b>{cur['neg']:.0f}%</b> negative."))
    return out


def render_signals(df, cur, prev):
    st.markdown('<div class="sec-title">Signals & takeaways</div>'
                '<div class="sec-cap">Auto-generated highlights for the current filters.</div>',
                unsafe_allow_html=True)
    if df.empty:
        st.info("No records for the current filters.")
        return
    items = "".join(
        f'<li><span class="dot" style="background:{c}"></span><span>{t}</span></li>'
        for c, t in build_signals(df, cur, prev)
    )
    st.markdown(f'<ul class="signal-list">{items}</ul>', unsafe_allow_html=True)


def _voice_cards(rows, accent):
    html = ""
    for _, r in rows.iterrows():
        text = (r["text"][:240] + "…") if len(r["text"]) > 240 else r["text"]
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        author = (r["author"] if isinstance(r["author"], str) and r["author"] else "anonymous")
        link = (f'<a href="{r["url"]}" target="_blank">open ↗</a>'
                if isinstance(r["url"], str) and r["url"].startswith("http") else "")
        html += (
            f'<div class="voice-card">'
            f'<div class="voice-meta">'
            f'<span><span class="pill" style="background:{accent}22;color:{accent}">'
            f'{r["sentiment_label"]}</span> &nbsp;{author}</span>'
            f'<span>👍 {int(r["engagement"]):,} · {r["sentiment_score"]:+.2f}</span></div>'
            f'<div class="voice-text">{text}</div>'
            f'<div style="margin-top:8px">{link}</div></div>'
        )
    return html


def render_voices(df):
    if df.empty:
        st.info("No records for the current filters.")
        return
    left, right = st.columns(2, gap="medium")
    pos = (df[df["sentiment_label"] == "positive"]
           .sort_values("engagement", ascending=False).head(5))
    neg = (df[df["sentiment_label"] == "negative"]
           .sort_values("engagement", ascending=False).head(5))
    with left:
        with st.container(border=True):
            st.markdown('<div class="sec-title" style="color:#2E9E5B">Top positive voices</div>'
                        '<div class="sec-cap">Most-engaged positive mentions.</div>',
                        unsafe_allow_html=True)
            if pos.empty:
                st.caption("No positive mentions in view.")
            else:
                st.markdown(_voice_cards(pos, GREEN), unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="sec-title" style="color:#D1495B">Top critical voices</div>'
                        '<div class="sec-cap">Most-engaged negative mentions.</div>',
                        unsafe_allow_html=True)
            if neg.empty:
                st.caption("No negative mentions in view.")
            else:
                st.markdown(_voice_cards(neg, RED), unsafe_allow_html=True)


def _source_kind(source) -> str:
    """Short 'comment' vs 'transcript' label for the aspect explorer cards."""
    return "transcript" if source == "youtube_transcript" else "comment"


# Cap the number of cards rendered per list so a busy aspect (e.g. scenery with
# ~1,800 mentions) stays snappy; the exact count is always shown in the caption.
EXPLORER_CAP = 200
EXPLORER_SPLIT_CAP = 150


def _comment_card(r) -> str:
    """One comment card: sentiment pill, date, source, likes, score, link.

    Long text collapses into a native <details> so each comment is expandable
    without leaving the scrollable list.
    """
    text = str(r["text"]).replace("<", "&lt;").replace(">", "&gt;")
    accent = SENTIMENT_COLORS.get(r["sentiment_label"], GREY)
    if len(text) > 240:
        body = (
            f'<details><summary>{text[:240]}… '
            f'<span class="more-link">show more</span></summary>'
            f'<div class="voice-text" style="margin-top:8px">{text}</div></details>'
        )
    else:
        body = f'<div class="voice-text">{text}</div>'
    date = str(r.get("date") or "")[:10]
    kind = _source_kind(r["source"])
    link = (f'<a href="{r["url"]}" target="_blank">open ↗</a>'
            if isinstance(r["url"], str) and r["url"].startswith("http") else "")
    return (
        f'<div class="voice-card">'
        f'<div class="voice-meta">'
        f'<span><span class="pill" style="background:{accent}22;color:{accent}">'
        f'{r["sentiment_label"]}</span> &nbsp;{date} · {kind}</span>'
        f'<span>👍 {int(r["engagement"]):,} · {r["sentiment_score"]:+.2f}</span></div>'
        f'{body}'
        f'<div style="margin-top:8px">{link}</div></div>'
    )


def _sort_comments(rows, sort_by):
    if sort_by == "Most positive":
        return rows.sort_values("sentiment_score", ascending=False)
    if sort_by == "Most negative":
        return rows.sort_values("sentiment_score", ascending=True)
    return rows.sort_values("engagement", ascending=False)  # Most engaged


def _comment_list(container, rows, cap):
    """Render up to `cap` comment cards into a scrollable container."""
    shown = rows.head(cap)
    container.markdown(
        "".join(_comment_card(r) for _, r in shown.iterrows()),
        unsafe_allow_html=True,
    )
    if len(rows) > cap:
        container.caption(f"Showing first {cap:,} of {len(rows):,} — narrow the "
                          f"filters or sort to see the rest.")


def render_aspect_explorer(df):
    st.markdown('<div class="sec-title">Explore comments by aspect</div>'
                '<div class="sec-cap">Pick an aspect to read the actual comments behind it. '
                'Respects every sidebar filter (country, dates, sentiment, source, '
                'relevant-only).</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("No records for the current filters.")
        return

    # Only offer aspects that actually appear in the filtered data.
    present = [
        a for a in ALL_ASPECTS
        if df["aspects"].str.contains(rf"(?:^|,){a}(?:,|$)", regex=True, na=False).any()
    ]
    if not present:
        st.info("No aspect-tagged comments for the current filters.")
        return

    c1, c2, c3 = st.columns([2.2, 1.2, 1.0], gap="medium")
    with c1:
        aspect = st.radio("Aspect", present, horizontal=True,
                          format_func=str.capitalize, key="explore_aspect")
    with c2:
        sort_by = st.selectbox(
            "Sort by", ["Most engaged", "Most positive", "Most negative"],
            key="explore_sort")
    with c3:
        split = st.toggle("Split + / −", value=True, key="explore_split",
                          help="Two columns: what people love vs criticise about this aspect.")

    pattern = rf"(?:^|,){aspect}(?:,|$)"
    sub = df[df["aspects"].str.contains(pattern, regex=True, na=False)]

    if split:
        left, right = st.columns(2, gap="medium")
        pos = _sort_comments(sub[sub["sentiment_label"] == "positive"], sort_by)
        neg = _sort_comments(sub[sub["sentiment_label"] == "negative"], sort_by)
        with left:
            st.markdown(
                f'<div class="sec-title" style="color:{GREEN}">Positive about '
                f'{aspect.capitalize()}</div>'
                f'<div class="sec-cap">{len(pos):,} comments</div>',
                unsafe_allow_html=True)
            box = st.container(height=520)
            if pos.empty:
                box.caption("None in view.")
            else:
                _comment_list(box, pos, EXPLORER_SPLIT_CAP)
        with right:
            st.markdown(
                f'<div class="sec-title" style="color:{RED}">Negative about '
                f'{aspect.capitalize()}</div>'
                f'<div class="sec-cap">{len(neg):,} comments</div>',
                unsafe_allow_html=True)
            box = st.container(height=520)
            if neg.empty:
                box.caption("None in view.")
            else:
                _comment_list(box, neg, EXPLORER_SPLIT_CAP)
    else:
        rows = _sort_comments(sub, sort_by)
        st.caption(f"{len(rows):,} comments tagged “{aspect}”")
        box = st.container(height=560)
        _comment_list(box, rows, EXPLORER_CAP)


# --------------------------------------------------------------------------- #
# Reviews tab (SerpApi star ratings + model-vs-stars validation)
# --------------------------------------------------------------------------- #
def _metric_card(label, value, sub=""):
    sub_html = f'<div class="metric-delta flat">{sub}</div>' if sub else ""
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>{sub_html}</div>')


def render_review_metrics(df):
    r = review_frame(df)
    n = len(r)
    avg = r["rating"].mean() if n else 0
    pct45 = 100 * (r["rating"] >= 4).mean() if n else 0
    pct12 = 100 * (r["rating"] <= 2).mean() if n else 0
    val = rating_vs_sentiment(df)
    sources = ", ".join(SOURCE_LABELS.get(s, s) for s in sorted(r["source"].unique()))
    cards = [
        ("Reviews in view", f"{n:,}", sources),
        ("Average rating", f"{avg:.1f} ★" if n else "—", f"{pct45:.0f}% rated 4–5★"),
        ("Model ↔ stars agree", f"{val['rate']:.0f}%" if val else "—",
         f"{val['agree']}/{val['total']} reviews match" if val else "no rated reviews"),
        ("Critical (1–2★)", f"{pct12:.0f}%" if n else "—", "share of low-star reviews"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(_metric_card(label, value, sub), unsafe_allow_html=True)


def render_rating_distribution(df):
    st.markdown('<div class="sec-title">Star rating distribution</div>'
                '<div class="sec-cap">How many reviews sit at each star level '
                '(green = 4–5★, grey = 3★, red = 1–2★).</div>', unsafe_allow_html=True)
    r = review_frame(df)
    if r.empty:
        st.info("No rated reviews in view.")
        return
    buckets = r["rating"].round().clip(1, 5).astype(int)
    counts = buckets.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    palette = {1: RED, 2: RED, 3: GREY, 4: GREEN, 5: GREEN}
    fig = go.Figure(go.Bar(
        x=[f"{s}★" for s in counts.index], y=[int(v) for v in counts.values],
        marker_color=[palette[s] for s in counts.index], marker_cornerradius=6,
        text=[int(v) for v in counts.values], textposition="outside",
        textfont=dict(color=INK, size=13),
        hovertemplate="%{x}: %{y} reviews<extra></extra>",
    ))
    style_fig(fig, height=300)
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_rating_agreement(df):
    st.markdown('<div class="sec-title">Model sentiment vs. star rating</div>'
                '<div class="sec-cap">Rows = sentiment the stars imply, columns = what our '
                'model predicted. The diagonal is agreement.</div>', unsafe_allow_html=True)
    val = rating_vs_sentiment(df)
    if not val:
        st.info("No rated reviews in view.")
        return
    conf = val["confusion"]
    labels = [s.capitalize() for s in SENTIMENTS]
    fig = go.Figure(go.Heatmap(
        z=conf.values, x=labels, y=labels,
        text=conf.values, texttemplate="%{text}", textfont=dict(size=15, color=INK),
        colorscale=[[0.0, "#F4F7F7"], [1.0, TEAL]], showscale=False, xgap=4, ygap=4,
        hovertemplate="stars imply %{y}<br>model said %{x}: %{z}<extra></extra>",
    ))
    style_fig(fig, height=300)
    fig.update_yaxes(autorange="reversed", title_text="Star-implied")
    fig.update_xaxes(title_text="Model prediction", side="bottom")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Agreement: {val['agree']} of {val['total']} rated reviews "
               f"({val['rate']:.0f}%). Models rarely emit ‘neutral’, so 3★ reviews "
               f"tend to land in the off-diagonal.")


def render_review_disagreements(df):
    st.markdown('<div class="sec-title">Where the model and the stars disagree</div>'
                '<div class="sec-cap">Reviews whose star rating and model sentiment point '
                'opposite ways — most confident first. Usually mixed reviews.</div>',
                unsafe_allow_html=True)
    val = rating_vs_sentiment(df)
    if not val:
        st.info("No rated reviews in view.")
        return
    dis = val["frame"]
    dis = dis[dis["expected"] != dis["sentiment_label"]].copy()
    if dis.empty:
        st.success("The model matches the stars on every review in view. ✓")
        return
    dis["absscore"] = dis["sentiment_score"].abs()
    dis = dis.sort_values("absscore", ascending=False).head(6)
    html = ""
    for _, row in dis.iterrows():
        stars = int(max(0, min(5, round(row["rating"]))))
        star_str = "★" * stars + "☆" * (5 - stars)
        star_color = GREEN if stars >= 4 else (RED if stars <= 2 else GREY)
        pred_color = SENTIMENT_COLORS.get(row["sentiment_label"], GREY)
        exp_color = SENTIMENT_COLORS.get(row["expected"], GREY)
        text = (row["text"][:240] + "…") if len(row["text"]) > 240 else row["text"]
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        link = (f'<a href="{row["url"]}" target="_blank">open ↗</a>'
                if isinstance(row["url"], str) and row["url"].startswith("http") else "")
        html += (
            f'<div class="voice-card">'
            f'<div class="voice-meta">'
            f'<span><span style="color:{star_color};font-size:13px">{star_str}</span>'
            f' &nbsp;stars imply <b style="color:{exp_color}">{row["expected"]}</b></span>'
            f'<span>model: <b style="color:{pred_color}">{row["sentiment_label"]}</b>'
            f' · {row["sentiment_score"]:+.2f}</span></div>'
            f'<div class="voice-text">{text}</div>'
            f'<div style="margin-top:8px">{link}</div></div>'
        )
    st.markdown(html, unsafe_allow_html=True)


def render_reviews(df):
    r = review_frame(df)
    if r.empty:
        st.info("No SerpApi review records for the current filters. Turn on the "
                "**Reviews (SerpApi)** data source in the sidebar (and widen the other "
                "filters) to see Google Hotels / Tripadvisor reviews and their ratings.")
        return
    render_review_metrics(df)
    st.write("")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            render_rating_distribution(df)
    with c2:
        with st.container(border=True):
            render_rating_agreement(df)
    with st.container(border=True):
        render_review_disagreements(df)


def render_table(df):
    st.markdown('<div class="sec-title">Browse the records</div>'
                '<div class="sec-cap">Every mention behind the charts above.</div>',
                unsafe_allow_html=True)
    search = st.text_input("Search text", placeholder="e.g. safari, Cape Town, expensive")
    view = df
    if search:
        view = view[view["text"].str.contains(search, case=False, na=False)]
    view = view.sort_values("timestamp", ascending=False)
    show = view[[
        "date", "source", "sentiment_label", "sentiment_score", "rating",
        "aspects", "emotion", "engagement", "text", "url",
    ]].rename(columns={"sentiment_label": "sentiment", "sentiment_score": "score"})
    st.caption(f"{len(show):,} records match")
    st.dataframe(
        show, use_container_width=True, hide_index=True, height=520,
        column_config={
            "url": st.column_config.LinkColumn("link", display_text="open"),
            "text": st.column_config.TextColumn("text", width="large"),
            "score": st.column_config.NumberColumn("score", format="%+.3f"),
            "rating": st.column_config.NumberColumn("rating", format="%.1f ★"),
            "source": st.column_config.TextColumn("source"),
        },
    )


# --------------------------------------------------------------------------- #
# Regional comparison render
# --------------------------------------------------------------------------- #
def render_compare_perception(metrics, cmap):
    st.markdown('<div class="sec-title">Perception score ranking</div>'
                '<div class="sec-cap">Higher = more positive overall sentiment '
                '(0–100).</div>', unsafe_allow_html=True)
    m = metrics.sort_values("perception")
    fig = go.Figure(go.Bar(
        y=m["country"], x=m["perception"], orientation="h",
        marker_color=[cmap[c] for c in m["country"]], marker_cornerradius=6,
        text=[f"{v}" for v in m["perception"]], textposition="outside",
        textfont=dict(color=INK, size=13),
        customdata=m["mentions"],
        hovertemplate="%{y}<br>perception: %{x}<br>%{customdata:,} mentions<extra></extra>",
    ))
    style_fig(fig, height=max(220, 60 * len(m)))
    fig.update_xaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_compare_mix(metrics):
    st.markdown('<div class="sec-title">Sentiment mix by country</div>'
                '<div class="sec-cap">Share of positive / neutral / negative within '
                'each country.</div>', unsafe_allow_html=True)
    m = metrics.sort_values("perception")
    fig = go.Figure()
    for s in SENTIMENTS:
        fig.add_bar(
            y=m["country"], x=m[s], orientation="h", name=s.capitalize(),
            marker_color=SENTIMENT_COLORS[s], marker_cornerradius=4,
            hovertemplate="%{y} — " + s + ": %{x:.0f}%%<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    style_fig(fig, height=max(220, 60 * len(m)))
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_compare_trend(df_all, granularity, cmap):
    st.markdown(f'<div class="sec-title">Perception over time by country</div>'
                f'<div class="sec-cap">Perception score per {granularity.lower()}, '
                f'one line per country.</div>', unsafe_allow_html=True)
    fig = go.Figure()
    plotted = False
    for country, g in df_all.groupby("country"):
        series = time_series(g, granularity)
        if series.empty:
            continue
        plotted = True
        fig.add_trace(go.Scatter(
            x=series["period"], y=series["perception"], name=country, mode="lines+markers",
            line=dict(color=cmap[country], width=3, shape="spline", smoothing=0.6),
            marker=dict(size=6),
            hovertemplate=country + "<br>%{x}: %{y:.0f}<extra></extra>",
        ))
    if not plotted:
        st.info("No records for the current filters.")
        return
    style_fig(fig, height=380)
    fig.update_yaxes(title_text="Perception", range=[0, 100])
    fig.add_hline(y=50, line_dash="dot", line_color=BORDER)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_compare_aspect_heatmap(df_all):
    st.markdown('<div class="sec-title">Net sentiment by aspect × country</div>'
                '<div class="sec-cap">Green = loved, red = criticised. Blank = no '
                'tagged mentions.</div>', unsafe_allow_html=True)
    mat = compare_aspect_net(df_all)
    if mat.empty:
        st.info("No aspect-tagged records for the current filters.")
        return
    aspects_present = [a for a in ALL_ASPECTS if a in mat.columns and mat[a].notna().any()]
    mat = mat[aspects_present]
    text = [[("" if pd.isna(v) else f"{v:+.0f}") for v in row] for row in mat.values]
    fig = go.Figure(go.Heatmap(
        z=mat.values, x=[a.capitalize() for a in mat.columns], y=mat.index.tolist(),
        zmid=0, zmin=-100, zmax=100,
        colorscale=[[0.0, RED], [0.5, "#F4F7F7"], [1.0, GREEN]],
        text=text, texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate="%{y} — %{x}: %{z:+.0f}%%<extra></extra>",
        xgap=3, ygap=3, colorbar=dict(title="net %", thickness=12),
    ))
    style_fig(fig, height=max(240, 70 * len(mat)))
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_compare_table(metrics):
    st.markdown('<div class="sec-title">Country scoreboard</div>'
                '<div class="sec-cap">Every country in view, ranked by perception.</div>',
                unsafe_allow_html=True)
    show = metrics.assign(
        rank=range(1, len(metrics) + 1),
    )[["rank", "country", "perception", "net", "positive", "negative",
       "mentions", "top_loved", "top_critical"]]
    st.dataframe(
        show, use_container_width=True, hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "perception": st.column_config.ProgressColumn(
                "perception", min_value=0, max_value=100, format="%d"),
            "net": st.column_config.NumberColumn("net %", format="%+.0f"),
            "positive": st.column_config.NumberColumn("pos %", format="%.0f"),
            "negative": st.column_config.NumberColumn("neg %", format="%.0f"),
            "mentions": st.column_config.NumberColumn("mentions", format="%,d"),
            "top_loved": st.column_config.TextColumn("most loved"),
            "top_critical": st.column_config.TextColumn("most criticised"),
        },
    )


def render_compare(df_all, granularity):
    countries = sorted(df_all["country"].dropna().unique().tolist())
    if not countries:
        st.info("No records for the current filters.")
        return
    picked = st.multiselect(
        "Countries to compare", countries, default=countries,
        help="Independent of the sidebar Country picker — choose any subset to compare.",
    )
    if not picked:
        st.info("Pick at least one country to compare.")
        return
    sub = df_all[df_all["country"].isin(picked)]
    cmap = country_color_map(countries)
    metrics = compare_metrics(sub)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            render_compare_perception(metrics, cmap)
    with c2:
        with st.container(border=True):
            render_compare_mix(metrics)
    with st.container(border=True):
        render_compare_trend(sub, granularity, cmap)
    with st.container(border=True):
        render_compare_aspect_heatmap(sub)
    with st.container(border=True):
        render_compare_table(metrics)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Africa Insights — Tourism Perception",
                       layout="wide", page_icon="🌍")
    inject_css()

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"Data file not found: {DATA_FILE}. Run `python run.py` first.")
        return

    # ----- Sidebar filters -----
    st.sidebar.header("Filters")
    countries = sorted(df["country"].dropna().unique().tolist())
    country = st.sidebar.selectbox("Country", countries, index=0)

    granularity = st.sidebar.radio("Date granularity", ["Year", "Month"], horizontal=True)
    year_range = st.sidebar.slider("Year range", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX))

    aspect_opts = [a for a in ALL_ASPECTS]
    aspects = st.sidebar.multiselect("Aspect (any of)", aspect_opts)
    sentiments = st.sidebar.multiselect("Sentiment", SENTIMENTS, default=SENTIMENTS)

    # Provider-level "Data source" filter: pick whole providers (YouTube,
    # Reviews, Reddit) and we expand them to their underlying sources.
    present_providers = [g for g in PROVIDER_GROUPS if df["provider"].eq(g).any()]
    extra_providers = sorted(set(df["provider"]) - set(PROVIDER_GROUPS))
    provider_opts = present_providers + extra_providers
    providers = st.sidebar.multiselect(
        "Data source", provider_opts, default=provider_opts,
        help="Filter by provider — YouTube, Reviews (Google Hotels / Tripadvisor "
             "via SerpApi), Reddit. Pick any mix, or leave all selected.",
    )
    sources = [s for p in providers for s in PROVIDER_GROUPS.get(p, [p])]
    kept_only = st.sidebar.toggle("Relevant records only", value=True,
                                  help="relevance_kept = True. Turn off to include everything.")

    common = dict(country=country, aspects=aspects, sentiments=sentiments,
                  sources=sources, kept_only=kept_only)
    filtered = filter_data(df, year_range=year_range, **common)

    # Same filters but across ALL countries — drives the regional comparison tab.
    filtered_all = filter_data(
        df, country="All", year_range=year_range, aspects=aspects,
        sentiments=sentiments, sources=sources, kept_only=kept_only,
    )

    # Previous, equally-sized window immediately before the selected range.
    lo, hi = year_range
    span = hi - lo + 1
    prev = filter_data(df, year_range=(lo - span, lo - 1), **common)
    cur_m, prev_m = summarize(filtered), summarize(prev)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Showing **{len(filtered):,}** of {len(df):,} records")
    st.sidebar.caption("The **Compare** tab spans all countries; the other tabs "
                       "follow the Country picker above.")

    # ----- Header + metric cards -----
    render_header(filtered, cur_m, prev_m)
    if filtered.empty:
        # Explain the empty view instead of showing a wall of zeros. The most
        # common dead-end: a Data source that has no rows for the chosen country
        # (e.g. Reviews (SerpApi) currently only cover South Africa).
        src_rows = df[df["source"].isin(sources)] if sources else df
        countries_with_src = sorted(src_rows["country"].dropna().unique().tolist())
        prov_label = " + ".join(providers) if providers else "the selected data sources"
        if countries_with_src and country not in countries_with_src:
            st.warning(
                f"No **{prov_label}** records for **{country}**. That data currently "
                f"exists for: **{', '.join(countries_with_src)}**. Switch the "
                f"**Country** picker at the top of the sidebar, or add another "
                f"**Data source**.", icon="⚠️")
        else:
            st.warning(
                "No records match the current filters. Try widening the year range, "
                "sentiment, or **Data source** in the sidebar.", icon="⚠️")
    else:
        render_metric_cards(cur_m, prev_m)
    st.write("")

    # The Reviews tab only appears when the dataset actually has review records.
    has_reviews = df["source"].isin(REVIEW_SOURCES).any()
    tab_names = ["Overview", "Compare", "Themes", "Voices"]
    if has_reviews:
        tab_names.append("Reviews")
    tab_names.append("Data")
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    with tabs["Overview"]:
        with st.container(border=True):
            render_ribbon(filtered, cur_m)
        with st.container(border=True):
            render_trend(filtered, granularity)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with st.container(border=True):
                render_source_comparison(filtered)
        with c2:
            with st.container(border=True):
                render_signals(filtered, cur_m, prev_m)

    with tabs["Compare"]:
        render_compare(filtered_all, granularity)

    with tabs["Themes"]:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with st.container(border=True):
                render_aspect_stack(filtered)
        with c2:
            with st.container(border=True):
                render_aspect_net(filtered)
        with st.container(border=True):
            render_aspect_explorer(filtered)

    with tabs["Voices"]:
        render_voices(filtered)

    if has_reviews:
        with tabs["Reviews"]:
            render_reviews(filtered)

    with tabs["Data"]:
        with st.container(border=True):
            render_table(filtered)


if __name__ == "__main__":
    main()
