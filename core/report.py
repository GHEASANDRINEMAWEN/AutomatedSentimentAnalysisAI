"""Branded PDF report for whatever the dashboard is currently showing.

`build_pdf(df, ...)` takes an ALREADY-FILTERED frame and returns PDF bytes, so
the report always describes exactly the view on screen — same country, dates,
aspects, sentiments and sources.

Two halves, both generated from the data (nothing is hard-coded):

  * charts    — sentiment breakdown, sentiment per aspect, trend over time and
                volume, drawn with matplotlib in the brand palette.
  * narrative — real sentences. `narrative()` computes the numbers and writes
                paragraphs about the mix, which themes are praised or criticised,
                which way the trend is moving, where the data came from and who
                the loudest voices are.

Deliberately free of Streamlit and of dashboard.py so it can be imported, run
and tested from a plain script against data/records.csv.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")           # headless: no display needed on a server
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

# --------------------------------------------------------------------------- #
# Brand tokens (mirrors the dashboard's design tokens)
# --------------------------------------------------------------------------- #
TEAL = "#127B82"        # brand accent
GREEN = "#2E9E5B"       # positive
GREY = "#9AA7A8"        # neutral
RED = "#D1495B"         # negative
INK = "#1C2B2E"         # headings
MUTED = "#5E7174"       # labels / captions
BORDER = "#E6EBEB"
FRAME = "#F5F7F8"

SENTIMENTS = ("positive", "neutral", "negative")
SENTIMENT_COLORS = {"positive": GREEN, "neutral": GREY, "negative": RED}

ALL_ASPECTS = (
    "food", "scenery", "safety", "wildlife", "hospitality", "transport", "cost",
    "weather", "electricity", "water", "housing",
)
SOURCE_LABELS = {
    "youtube": "YouTube comments",
    "youtube_transcript": "YouTube transcript segments",
    "google_hotels": "Google Hotels reviews",
    "tripadvisor": "Tripadvisor reviews",
    "reddit": "Reddit posts",
}

# Aspects need a net-sentiment margin this wide before the narrative calls one
# "praised" or "criticised" — below it the split is a coin-flip, not a finding.
NET_MARGIN = 5.0
# An aspect also needs this many mentions before it can headline a sentence.
MIN_ASPECT_MENTIONS = 15
# Perception must move at least this many points to count as a real trend.
TREND_MARGIN = 2.0
# Below this many records the view is too thin for theme or trend claims. The
# report still renders — it says so plainly instead of dressing up noise as a
# finding ("food is the most loved aspect, +100 net across 1 mention").
MIN_SAMPLE = 50


# --------------------------------------------------------------------------- #
# Aggregation (plain pandas — no dashboard/Streamlit dependency)
# --------------------------------------------------------------------------- #
def summarize(df) -> dict:
    """Headline metrics for a frame."""
    n = len(df)
    counts = df["sentiment_label"].value_counts().to_dict() if n else {}
    split = {s: int(counts.get(s, 0)) for s in SENTIMENTS}
    pct = {s: (100 * split[s] / n if n else 0.0) for s in SENTIMENTS}
    perception = round((df["sentiment_score"].mean() + 1) / 2 * 100) if n else 0
    return dict(n=n, split=split, pos=pct["positive"], neu=pct["neutral"],
                neg=pct["negative"], net=pct["positive"] - pct["negative"],
                perception=int(perception))


def aspect_table(df) -> pd.DataFrame:
    """Per-aspect mention count and positive/neutral/negative percentages."""
    empty = pd.DataFrame(
        columns=["aspect", "total", "positive", "neutral", "negative", "net"])
    if df.empty:
        return empty
    exploded = df.assign(aspect=df["aspects"].str.split(",")).explode("aspect")
    exploded = exploded[exploded["aspect"].isin(ALL_ASPECTS)]
    if exploded.empty:
        return empty
    grouped = (exploded.groupby(["aspect", "sentiment_label"]).size()
               .unstack(fill_value=0))
    for s in SENTIMENTS:
        if s not in grouped:
            grouped[s] = 0
    grouped["total"] = grouped[list(SENTIMENTS)].sum(axis=1)
    out = grouped.reset_index()
    for s in SENTIMENTS:
        out[s] = 100 * out[s] / out["total"]
    out["net"] = out["positive"] - out["negative"]
    return out[["aspect", "total", *SENTIMENTS, "net"]].sort_values(
        "net", ascending=False).reset_index(drop=True)


def time_series(df, granularity: str = "Year") -> pd.DataFrame:
    """Perception score and record volume per period."""
    key = "year" if granularity == "Year" else "month"
    if df.empty or key not in df.columns:
        return pd.DataFrame(columns=["period", "perception", "volume", "neg_pct"])
    grouped = (df.groupby(key)
               .agg(avg=("sentiment_score", "mean"), volume=("source", "size"))
               .reset_index().rename(columns={key: "period"}))
    neg = (df["sentiment_label"].eq("negative").groupby(df[key]).mean() * 100)
    grouped["neg_pct"] = grouped["period"].map(neg)
    grouped["perception"] = (grouped["avg"] + 1) / 2 * 100
    grouped["period"] = grouped["period"].astype(str)
    return grouped.sort_values("period").reset_index(drop=True)


def split_halves(df):
    """Split the frame into earlier and later halves of its own date range.

    Used to say whether a theme's criticism is rising or falling WITHIN the
    selected window, without needing data from outside the user's filters.
    """
    if df.empty or "timestamp" not in df.columns:
        return df.iloc[0:0], df.iloc[0:0]
    ordered = df.sort_values("timestamp")
    midpoint = len(ordered) // 2
    if midpoint == 0:
        return ordered.iloc[0:0], ordered
    return ordered.iloc[:midpoint], ordered.iloc[midpoint:]


def date_span(df):
    """(earliest, latest) timestamps in the frame, or (None, None)."""
    if df.empty or "timestamp" not in df.columns:
        return None, None
    stamps = df["timestamp"].dropna()
    if stamps.empty:
        return None, None
    return stamps.min(), stamps.max()


def _month_name(ts) -> str:
    return ts.strftime("%B %Y") if ts is not None else "n/a"


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _style_axes(ax, *, xgrid=False, ygrid=False):
    """Recessive axes: no box, hairline grid, muted tick labels."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_axisbelow(True)
    if xgrid:
        ax.xaxis.grid(True, color=BORDER, linewidth=0.8)
    if ygrid:
        ax.yaxis.grid(True, color=BORDER, linewidth=0.8)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return buf.getvalue()


def chart_sentiment_breakdown(metrics) -> bytes:
    """One 100%-stacked bar: the positive / neutral / negative mix.

    Every segment is labelled with its own percentage, so the split is readable
    without relying on the colour (green/red is the classic colour-blind trap).
    """
    fig, ax = plt.subplots(figsize=(7.2, 1.5))
    left = 0.0
    for label in SENTIMENTS:
        width = getattr_pct(metrics, label)
        if width <= 0:
            continue
        ax.barh([0], [width], left=[left], height=0.5,
                color=SENTIMENT_COLORS[label], edgecolor="white", linewidth=1.5)
        if width >= 7:      # only label a segment wide enough to hold the text
            ax.text(left + width / 2, 0,
                    f"{label}\n{width:.0f}%  ·  {metrics['split'][label]:,}",
                    ha="center", va="center", color="white", fontsize=8.5,
                    fontweight="bold", linespacing=1.4)
        left += width
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    _style_axes(ax)
    return _png(fig)


def getattr_pct(metrics, label) -> float:
    return {"positive": metrics["pos"], "neutral": metrics["neu"],
            "negative": metrics["neg"]}[label]


def chart_aspect_sentiment(table: pd.DataFrame) -> bytes:
    """100%-stacked bars per aspect, best-regarded theme at the top."""
    rows = table.head(11).iloc[::-1]          # barh draws bottom-up
    height = max(2.0, 0.42 * len(rows) + 0.9)
    fig, ax = plt.subplots(figsize=(7.2, height))
    ypos = range(len(rows))
    left = [0.0] * len(rows)
    for label in SENTIMENTS:
        widths = rows[label].tolist()
        ax.barh(list(ypos), widths, left=left, height=0.6,
                color=SENTIMENT_COLORS[label], edgecolor="white", linewidth=1.2,
                label=label)
        for i, (w, l) in enumerate(zip(widths, left)):
            if w >= 12:
                ax.text(l + w / 2, i, f"{w:.0f}%", ha="center", va="center",
                        color="white", fontsize=7.5, fontweight="bold")
        left = [a + b for a, b in zip(left, widths)]
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([f"{a.capitalize()}  ({int(t):,})"
                        for a, t in zip(rows["aspect"], rows["total"])],
                       fontsize=8.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    _style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10 - 0.12 / height),
              ncol=3, frameon=False, fontsize=8.5, labelcolor=MUTED,
              handlelength=1.2, handleheight=1.0, columnspacing=2.0)
    return _png(fig)


def chart_trend(series: pd.DataFrame, granularity: str) -> bytes:
    """Perception score over time — one measure, one axis, one line."""
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    x = list(range(len(series)))
    ax.axhline(50, color=BORDER, linewidth=1.2, zorder=1)
    ax.text(len(x) - 0.9 if x else 0, 50.8, "neutral (50)", fontsize=7,
            color=MUTED, ha="right", va="bottom")
    ax.plot(x, series["perception"], color=TEAL, linewidth=2.0, zorder=3)
    ax.fill_between(x, 50, series["perception"], color=TEAL, alpha=0.10, zorder=2)
    ax.scatter(x, series["perception"], s=26, color=TEAL, zorder=4,
               edgecolor="white", linewidth=1.2)
    # Label only the endpoints, so the line stays readable.
    for i in ({0, len(x) - 1} if x else set()):
        ax.annotate(f"{series['perception'].iloc[i]:.0f}", (i, series["perception"].iloc[i]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(series["period"], rotation=45 if granularity == "Month" else 0,
                       ha="right" if granularity == "Month" else "center")
    ax.set_ylabel("perception score", fontsize=8, color=MUTED)
    _style_axes(ax, ygrid=True)
    return _png(fig)


def chart_volume(series: pd.DataFrame, granularity: str) -> bytes:
    """Record volume per period — how much conversation each period carried."""
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    x = list(range(len(series)))
    ax.bar(x, series["volume"], width=0.62, color=TEAL, alpha=0.85)
    if x:
        peak = int(series["volume"].idxmax())
        ax.annotate(f"{int(series['volume'].iloc[peak]):,}",
                    (peak, series["volume"].iloc[peak]),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(series["period"], rotation=45 if granularity == "Month" else 0,
                       ha="right" if granularity == "Month" else "center")
    ax.set_ylabel("mentions", fontsize=8, color=MUTED)
    _style_axes(ax, ygrid=True)
    return _png(fig)


# --------------------------------------------------------------------------- #
# Narrative — real sentences, computed from the filtered data
# --------------------------------------------------------------------------- #
def _join(items) -> str:
    """'a', 'a and b', 'a, b and c'."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f'{", ".join(items[:-1])} and {items[-1]}'


def _count(value, noun: str = "mention") -> str:
    """'1 mention' / '12 mentions' — never the '1 mentions' of naive f-strings."""
    value = int(value)
    return f"{value:,} {noun}{'' if value == 1 else 's'}"


def _aspect_phrase(row) -> str:
    return f"{row['aspect']} ({row['net']:+.0f} net across {_count(row['total'])})"


def narrative(df, *, country: str, granularity: str = "Year") -> list:
    """Write the findings as sentences. Returns a list of (heading, text)."""
    metrics = summarize(df)
    n = metrics["n"]
    if not n:
        return [("Summary", "No records match the current filters, so there is "
                            "nothing to report. Widen the year range, sentiment "
                            "or data-source filters and generate the report again.")]

    lo, hi = date_span(df)
    where = country if country and country != "All" else "the countries in view"
    when = (f"between {_month_name(lo)} and {_month_name(hi)}"
            if lo is not None and hi is not None else "across the dataset")
    out = []

    # --- 1. The headline mix -------------------------------------------------
    tone = ("strongly positive" if metrics["net"] >= 40 else
            "positive" if metrics["net"] >= 15 else
            "mixed but net-positive" if metrics["net"] > 0 else
            "mixed but net-negative" if metrics["net"] > -15 else "negative")
    thin = n < MIN_SAMPLE
    para = (f"Across {_count(n)} of {where} {when}, sentiment was "
            f"{metrics['pos']:.0f}% positive, {metrics['neu']:.0f}% neutral and "
            f"{metrics['neg']:.0f}% negative — a net sentiment of "
            f"{metrics['net']:+.0f} points, which reads as {tone}. On the "
            f"0–100 perception scale the period scores "
            f"<b>{metrics['perception']}</b>.")
    if thin:
        para += (f" This is a thin view: {_count(n, 'record')} is below the "
                 f"{MIN_SAMPLE}-record threshold this report uses for reliable "
                 f"reading, so treat every figure below as indicative only.")
    out.append(("What the data says", para))

    # --- 2. What is praised, what is criticised ------------------------------
    table = aspect_table(df)
    solid = table[table["total"] >= MIN_ASPECT_MENTIONS]
    if not table.empty:
        busiest = table.sort_values("total", ascending=False).iloc[0]
        bits = []
        if solid.empty:
            # Every theme is below the mention floor — say that, rather than
            # promoting a one-off comment into a headline finding.
            bits.append(
                f"No theme reaches the {MIN_ASPECT_MENTIONS}-mention floor "
                f"needed for a reliable read. The most-discussed is "
                f"{busiest['aspect']}, raised in {_count(busiest['total'])} "
                f"({busiest['positive']:.0f}% of them positive) — indicative, "
                f"not conclusive.")
        else:
            loved = solid[solid["net"] >= NET_MARGIN].head(3)
            disliked = solid[solid["net"] <= -NET_MARGIN].tail(3).iloc[::-1]
            if not loved.empty:
                bits.append("Visitors were most positive about "
                            + _join(_aspect_phrase(r) for _, r in loved.iterrows())
                            + ".")
            if not disliked.empty:
                bits.append("Criticism concentrated on "
                            + _join(_aspect_phrase(r) for _, r in disliked.iterrows())
                            + ".")
            elif len(solid) > 1:
                # Nothing clears the margin, but the reader still needs to know
                # where the weakest reception is — hedged as the numbers demand.
                weakest = solid.tail(2).iloc[::-1]
                bits.append("No theme is decisively negative; the coolest "
                            "reception went to "
                            + _join(_aspect_phrase(r) for _, r in weakest.iterrows())
                            + ".")
            if loved.empty and disliked.empty:
                bits.append("Overall no single theme stands out as clearly "
                            "praised or clearly criticised — every aspect sits "
                            "close to an even split.")
            bits.append(f"{busiest['aspect'].capitalize()} was the most discussed "
                        f"theme overall, raised in {_count(busiest['total'])} "
                        f"({busiest['positive']:.0f}% of them positive).")
        out.append(("Themes behind the score", " ".join(bits)))

    # --- 3. Which way it is moving -------------------------------------------
    series = time_series(df, granularity)
    bits = []
    if thin:
        bits.append(f"With only {_count(n, 'record')} spread over "
                    f"{_count(len(series), 'period')}, there is not enough "
                    f"volume to read a direction; the trend chart is shown for "
                    f"completeness. Collect more coverage before acting on it.")
    elif len(series) >= 2:
        first, last = series.iloc[0], series.iloc[-1]
        delta = last["perception"] - first["perception"]
        direction = ("improved" if delta >= TREND_MARGIN else
                     "declined" if delta <= -TREND_MARGIN else "held steady")
        bits.append(
            f"Perception has {direction} over the period: it moved from "
            f"{first['perception']:.0f} in {first['period']} to "
            f"{last['perception']:.0f} in {last['period']} ({delta:+.0f} points).")
        peak = series.loc[series["volume"].idxmax()]
        bits.append(f"Conversation volume peaked in {peak['period']} with "
                    f"{_count(peak['volume'])}, out of {n:,} in total.")
    else:
        bits.append(f"All {_count(n)} fall in a single period, so there is no "
                    f"trend to read yet.")

    # Rising / falling criticism per theme, comparing the two halves of the window.
    earlier, later = split_halves(df)
    if not thin and len(earlier) >= MIN_ASPECT_MENTIONS and len(later) >= MIN_ASPECT_MENTIONS:
        before = aspect_table(earlier).set_index("aspect")
        after = aspect_table(later).set_index("aspect")
        shared = [a for a in after.index
                  if a in before.index and after.loc[a, "total"] >= MIN_ASPECT_MENTIONS]
        shifts = sorted(
            ((a, after.loc[a, "negative"] - before.loc[a, "negative"]) for a in shared),
            key=lambda kv: kv[1])
        if shifts:
            worst, worst_shift = shifts[-1]
            best, best_shift = shifts[0]
            if worst_shift >= NET_MARGIN:
                bits.append(f"Negative sentiment around {worst} rose "
                            f"{worst_shift:+.0f} points in the second half of the "
                            f"period compared with the first.")
            if best_shift <= -NET_MARGIN:
                bits.append(f"Criticism of {best} eased over the same span "
                            f"({best_shift:+.0f} points of negative share).")
    out.append(("Direction of travel", " ".join(bits)))

    # --- 4. Where the evidence comes from ------------------------------------
    counts = df["source"].value_counts()
    parts = [f"{int(c):,} {SOURCE_LABELS.get(s, s)}" for s, c in counts.items()]
    bits = [f"The picture is drawn from {_join(parts)}."]
    if "country_source" in df.columns:
        content = int(df["country_source"].isin(["content", "context"]).sum())
        bits.append(f"{100 * content / n:.0f}% of these records were attributed to "
                    f"{where} from what the text itself is about, rather than from "
                    f"the search query that surfaced them.")
    emotive = df.loc[df["emotion"].ne("neutral") & df["emotion"].ne(""), "emotion"]
    if not emotive.empty:
        top = emotive.value_counts()
        bits.append(f"Beyond neutral, <b>{top.index[0]}</b> is the dominant "
                    f"emotional register, tagged on {_count(top.iloc[0])} "
                    f"({100 * top.iloc[0] / n:.0f}% of the total).")
    out.append(("Where the evidence comes from", " ".join(bits)))
    return out


def top_voices(df, limit: int = 3):
    """Most-engaged positive and negative mentions: [(label, row), ...]."""
    if df.empty:
        return []
    picks = []
    for label in ("positive", "negative"):
        rows = (df[df["sentiment_label"] == label]
                .sort_values("engagement", ascending=False).head(limit))
        picks.extend((label, row) for _, row in rows.iterrows())
    return picks


# --------------------------------------------------------------------------- #
# PDF assembly
# --------------------------------------------------------------------------- #
def _safe(text: str, limit: int = 400) -> str:
    """Make arbitrary comment text safe for reportlab's Latin-1 base fonts.

    Comments are full of emoji and non-Latin scripts, which the built-in
    Helvetica cannot draw (they come out as black boxes). Drop what the font
    cannot render, collapse whitespace, escape XML, and trim.
    """
    text = " ".join(str(text or "").split())
    text = "".join(ch for ch in text if ch == "\n" or 32 <= ord(ch) < 256)
    text = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text or "(no renderable text)"


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=21, leading=25, textColor=colors.white,
                                alignment=0, spaceAfter=2),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=9.5,
                                   leading=13.5, textColor=colors.HexColor("#CFE6E7")),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=12.5, leading=16, textColor=colors.HexColor(TEAL),
                             spaceBefore=12, spaceAfter=5),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.8, leading=15,
                               textColor=colors.HexColor(INK), alignment=TA_JUSTIFY),
        "caption": ParagraphStyle("c", parent=base["Normal"], fontSize=8.2, leading=11.5,
                                  textColor=colors.HexColor(MUTED), spaceAfter=4),
        "quote": ParagraphStyle("q", parent=base["Normal"], fontSize=9, leading=13,
                                textColor=colors.HexColor(INK), leftIndent=6),
        "meta": ParagraphStyle("m", parent=base["Normal"], fontSize=7.6, leading=10.5,
                               textColor=colors.HexColor(MUTED)),
    }


def _header_band(title, subtitle, styles, width):
    """Teal masthead carrying the country, date range and record count."""
    inner = [[Paragraph(title, styles["title"])],
             [Paragraph(subtitle, styles["subtitle"])]]
    band = Table(inner, colWidths=[width])
    band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(TEAL)),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (0, 0), 13),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 13),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return band


def _metric_row(metrics, styles, width):
    """Five at-a-glance tiles: volume, the sentiment mix, net and perception."""
    cells = [
        ("Records in view", f"{metrics['n']:,}", INK),
        ("Positive", f"{metrics['pos']:.0f}%", GREEN),
        ("Neutral", f"{metrics['neu']:.0f}%", MUTED),
        ("Negative", f"{metrics['neg']:.0f}%", RED),
        ("Perception", f"{metrics['perception']}/100", TEAL),
    ]
    label_style = ParagraphStyle("ml", fontName="Helvetica", fontSize=7.4,
                                 leading=9, textColor=colors.HexColor(MUTED),
                                 alignment=1)
    row_labels, row_values = [], []
    for label, value, color in cells:
        row_labels.append(Paragraph(label.upper(), label_style))
        row_values.append(Paragraph(
            value, ParagraphStyle("mv", fontName="Helvetica-Bold", fontSize=16,
                                  leading=19, textColor=colors.HexColor(color),
                                  alignment=1)))
    table = Table([row_values, row_labels], colWidths=[width / len(cells)] * len(cells))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(FRAME)),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(BORDER)),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor(BORDER)),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _image(png: bytes, width):
    """Scale a chart PNG to the frame width, preserving its aspect ratio."""
    reader = io.BytesIO(png)
    img = Image(reader)
    img.drawHeight = width * img.imageHeight / img.imageWidth
    img.drawWidth = width
    return img


def _voice_block(picks, styles, width):
    """Quote cards for the most-engaged supportive and critical mentions."""
    rows = []
    for label, r in picks:
        accent = GREEN if label == "positive" else RED
        author = _safe(r.get("author") or "anonymous", 40)
        when = ""
        if pd.notna(r.get("timestamp")):
            when = f" · {pd.Timestamp(r['timestamp']).strftime('%b %Y')}"
        meta = (f'<font color="{accent}"><b>{label.upper()}</b></font> · '
                f'{author}{when} · {int(r.get("engagement") or 0):,} likes · '
                f'score {float(r.get("sentiment_score") or 0):+.2f}')
        rows.append([Paragraph(
            f'<font size="7.4" color="{MUTED}">{meta}</font><br/>'
            f'“{_safe(r.get("text"), 300)}”', styles["quote"])])
    table = Table(rows, colWidths=[width])
    style = [
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(FRAME)),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(BORDER)),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor(BORDER)),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    table.setStyle(TableStyle(style))
    return table


def _filters_table(filters, styles, width):
    rows = [[Paragraph(f"<b>{k}</b>", styles["meta"]), Paragraph(str(v), styles["meta"])]
            for k, v in filters.items()]
    table = Table(rows, colWidths=[width * 0.28, width * 0.72])
    table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(BORDER)),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor(BORDER)),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _decorate(canvas, doc):
    """Teal rule under the top margin and a muted footer with the page number."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor(BORDER))
    canvas.setLineWidth(0.6)
    canvas.line(doc.leftMargin, 14 * mm, doc.leftMargin + doc.width, 14 * mm)
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(doc.leftMargin, 10 * mm,
                      "Africa Insights — Tourism Perception")
    canvas.drawRightString(doc.leftMargin + doc.width, 10 * mm,
                           f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def report_title(country: str) -> str:
    where = country if country and country != "All" else "All countries"
    return f"{where} — Tourism Perception Report"


def build_pdf(df, *, country: str = "All", granularity: str = "Year",
              filters: dict | None = None) -> bytes:
    """Render the filtered frame as a branded PDF and return its bytes.

    `df` must already have the dashboard's filters applied — the report describes
    exactly what it is given. `filters` is echoed into the methodology appendix so
    a reader can see which view produced the numbers.
    """
    metrics = summarize(df)
    lo, hi = date_span(df)
    span = (f"{_month_name(lo)} – {_month_name(hi)}"
            if lo is not None else "no dated records")
    generated = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, title=report_title(country),
        author="Africa Insights", subject="Tourism perception analysis",
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=13 * mm, bottomMargin=20 * mm,
    )
    styles = _styles()
    width = doc.width
    story = []

    story.append(_header_band(
        report_title(country),
        f"{span} &nbsp;·&nbsp; {metrics['n']:,} records in view "
        f"&nbsp;·&nbsp; generated {generated}",
        styles, width))
    story.append(Spacer(1, 10))
    story.append(_metric_row(metrics, styles, width))
    story.append(Spacer(1, 6))

    # --- Narrative -----------------------------------------------------------
    for heading, text in narrative(df, country=country, granularity=granularity):
        story.append(Paragraph(heading, styles["h2"]))
        story.append(Paragraph(text, styles["body"]))

    if metrics["n"] == 0:
        doc.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
        return buf.getvalue()

    # --- Charts --------------------------------------------------------------
    table = aspect_table(df)
    series = time_series(df, granularity)

    def figure(heading, caption, png):
        """Heading + caption + chart as one unbreakable block, so a heading can
        never be stranded at the foot of a page without its chart."""
        return KeepTogether([Paragraph(heading, styles["h2"]),
                             Paragraph(caption, styles["caption"]),
                             _image(png, width)])

    story.append(figure(
        "Sentiment breakdown",
        f"How the {metrics['n']:,} records in view split across sentiment "
        f"classes. Net sentiment is {metrics['net']:+.0f} points.",
        chart_sentiment_breakdown(metrics)))

    if not table.empty:
        story.append(PageBreak())
        story.append(figure(
            "Sentiment per aspect",
            "Each travel theme mentioned in the view, ranked by net sentiment "
            "(most positively regarded first). Mention counts in brackets.",
            chart_aspect_sentiment(table)))

    if not series.empty:
        # The aspect chart is tall enough that trend + volume never both fit
        # under it; giving the time charts their own page beats leaving one of
        # them alone on a mostly-blank page.
        story.append(PageBreak())
        story.append(figure(
            f"Trend over time (by {granularity.lower()})",
            "Perception score per period on a 0–100 scale, where 50 is neutral.",
            chart_trend(series, granularity)))
        story.append(figure(
            f"Volume (by {granularity.lower()})",
            "Number of records per period — how much of the conversation each "
            "period carries. Read the trend above against this.",
            chart_volume(series, granularity)))

    # --- Voices --------------------------------------------------------------
    picks = top_voices(df)
    if picks:
        story.append(PageBreak())
        story.append(Paragraph("The loudest voices", styles["h2"]))
        story.append(Paragraph(
            "The most-engaged supportive and critical mentions in this view. "
            "Emoji and non-Latin characters are stripped for print.",
            styles["caption"]))
        story.append(_voice_block(picks, styles, width))

    # --- Methodology ---------------------------------------------------------
    story.append(Paragraph("How this view was built", styles["h2"]))
    active = dict(filters or {})
    active.setdefault("Country", country)
    active.setdefault("Date range", span)
    active.setdefault("Records", f"{metrics['n']:,}")
    story.append(KeepTogether(_filters_table(active, styles, width)))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Sentiment is scored by a multilingual transformer; aspects and emotion "
        "are rule-based taggers; each record is attributed to the country its "
        "text is about rather than the search query that surfaced it. Every "
        "number and sentence above is computed from the filtered records.",
        styles["meta"]))

    doc.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    return buf.getvalue()
