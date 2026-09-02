"""Intelligence-grade PDF report in the Africa INSIGHTS house style.

`build_pdf(df, ...)` takes an ALREADY-FILTERED frame and returns PDF bytes, so
the report always describes exactly the view on screen — same country, dates,
aspects, sentiments, sources and segments.

Three layers, and the separation between them is the point:

  * `core.facts`     computes every figure from the filtered records.
  * `core.narrative` writes the analysis around those figures — Claude when a key
                     is available, a deterministic template otherwise — and
                     verifies that every number in the finished prose came from
                     the fact pack.
  * this module      lays the result out: title block, seven numbered sections,
                     four charts, quoted voices, DATA NOTE callouts, methodology.

No figure is computed here and none is computed by the model. If a number is on
the page, `core.facts` derived it from the records.

House style
-----------
Title block: country, period, and the research-team byline on near-black.
Sections are numbered 01-07 with a gold rule. Body copy is analytical prose;
DATA NOTE callouts mark every place the evidence is too thin to carry a claim.

Deliberately free of Streamlit and of dashboard.py so it can be imported, run
and tested from a plain script against data/records.csv.
"""

from __future__ import annotations

import io

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

from core import facts, narrative as narrative_engine, rails

# --------------------------------------------------------------------------- #
# Brand tokens
# --------------------------------------------------------------------------- #
INK = "#0D0D0D"          # title block, headings
INK_SOFT = "#1A1A1A"     # body copy, secondary panels
GOLD = "#FECF2F"         # accent: section rules, highlights, volume bars
GOLD_LIGHT = "#FEE97A"   # accent, softened
GREEN = "#6FCF97"        # positive
RED = "#EB5757"          # negative
GREY = "#6B6B6B"         # neutral, labels, captions
CREAM = "#F5F5E8"        # section backgrounds and callouts
WHITE = "#FFFFFF"
HAIRLINE = "#E3E3D6"     # borders, drawn from the cream family

SENTIMENTS = facts.SENTIMENTS
SENTIMENT_COLORS = {"positive": GREEN, "neutral": GREY, "negative": RED}
ALL_ASPECTS = facts.ALL_ASPECTS
SOURCE_LABELS = facts.SOURCE_LABELS

BYLINE = ("Prepared by Africa INSIGHTS Research Team &nbsp;|&nbsp; "
          "Vertical: Travel &amp; Tourism")

# Honesty rails live in core/rails.py so the dashboard, the fact pack and this
# report can never disagree about what counts as enough evidence.
NET_MARGIN = rails.NET_MARGIN
MIN_ASPECT_MENTIONS = rails.MIN_ASPECT_MENTIONS
TREND_MARGIN = rails.TREND_MARGIN
MIN_SAMPLE = rails.MIN_SAMPLE

# Re-exported so existing callers (dashboard, validate_report) keep working
# against the aggregation helpers they already import from here.
summarize = facts.summarize
aspect_table = facts.aspect_table
time_series = facts.time_series
split_halves = facts.split_halves
date_span = facts.date_span


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
    ax.tick_params(colors=GREY, labelsize=8, length=0)
    ax.set_axisbelow(True)
    if xgrid:
        ax.xaxis.grid(True, color=HAIRLINE, linewidth=0.8)
    if ygrid:
        ax.yaxis.grid(True, color=HAIRLINE, linewidth=0.8)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return buf.getvalue()


def chart_sentiment_breakdown(sentiment: dict) -> bytes:
    """One 100%-stacked bar: the positive / neutral / negative mix.

    Every segment carries its own percentage, so the split survives being read
    in greyscale or by someone who cannot separate the green from the red.
    """
    widths = {"positive": sentiment["positive_pct"],
              "neutral": sentiment["neutral_pct"],
              "negative": sentiment["negative_pct"]}
    counts = {"positive": sentiment["positive_records"],
              "neutral": sentiment["neutral_records"],
              "negative": sentiment["negative_records"]}
    fig, ax = plt.subplots(figsize=(7.2, 1.55))
    left = 0.0
    for label in SENTIMENTS:
        width = widths[label] or 0.0
        if width <= 0:
            continue
        ax.barh([0], [width], left=[left], height=0.52,
                color=SENTIMENT_COLORS[label], edgecolor="white", linewidth=1.5)
        if width >= 7:      # only label a segment wide enough to hold the text
            ax.text(left + width / 2, 0,
                    f"{label}\n{facts.pct(width)}  ·  {counts[label]:,}",
                    ha="center", va="center",
                    color=INK if label != "neutral" else "white",
                    fontsize=8.5, fontweight="bold", linespacing=1.4)
        left += width
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    _style_axes(ax)
    return _png(fig)


def chart_aspect_sentiment(aspects: list) -> bytes:
    """100%-stacked bars per theme, best-regarded at the top.

    Themes below the mention floor are drawn faded, so the reader can see the
    whole picture without mistaking a three-comment theme for a finding.
    """
    rows = list(aspects)[:11][::-1]           # barh draws bottom-up
    height = max(2.0, 0.42 * len(rows) + 0.9)
    fig, ax = plt.subplots(figsize=(7.2, height))
    ypos = range(len(rows))
    left = [0.0] * len(rows)
    for label in SENTIMENTS:
        widths = [r[f"{label}_pct"] or 0.0 for r in rows]
        ax.barh(list(ypos), widths, left=left, height=0.6,
                color=SENTIMENT_COLORS[label], edgecolor="white", linewidth=1.2,
                label=label,
                alpha=1.0)
        for i, (w, l, row) in enumerate(zip(widths, left, rows)):
            if w >= 12:
                ax.text(l + w / 2, i, facts.pct(w), ha="center", va="center",
                        color=INK if label != "neutral" else "white",
                        fontsize=7.5, fontweight="bold",
                        alpha=1.0 if row["reportable"] else 0.55)
        left = [a + b for a, b in zip(left, widths)]
    # Fade the whole row for a theme that cannot headline a finding.
    for i, row in enumerate(rows):
        if not row["reportable"]:
            ax.barh([i], [100], height=0.6, color="white", alpha=0.45, zorder=5)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(
        [f"{r['aspect'].capitalize()}  ({r['mentions']:,})"
         + ("" if r["reportable"] else "  ·  thin")
         for r in rows], fontsize=8.5, color=INK)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if not row["reportable"]:
            tick.set_color(GREY)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    _style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10 - 0.12 / height),
              ncol=3, frameon=False, fontsize=8.5, labelcolor=GREY,
              handlelength=1.2, handleheight=1.0, columnspacing=2.0)
    return _png(fig)


def chart_trend(periods: list, granularity: str) -> bytes:
    """Perception score over time — one measure, one axis, one line."""
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    x = list(range(len(periods)))
    scores = [p["perception_score"] or 0 for p in periods]
    labels = [p["period"] for p in periods]
    ax.axhline(50, color=HAIRLINE, linewidth=1.2, zorder=1)
    ax.text(len(x) - 0.9 if x else 0, 50.8, "neutral (50)", fontsize=7,
            color=GREY, ha="right", va="bottom")
    ax.fill_between(x, 50, scores, color=GOLD, alpha=0.30, zorder=2)
    ax.plot(x, scores, color=INK, linewidth=2.0, zorder=3)
    ax.scatter(x, scores, s=30, color=GOLD, zorder=4, edgecolor=INK, linewidth=1.1)
    # Label only the endpoints, so the line stays readable.
    for i in ({0, len(x) - 1} if x else set()):
        ax.annotate(f"{scores[i]:.0f}", (i, scores[i]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color=INK,
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if granularity == "month" else 0,
                       ha="right" if granularity == "month" else "center")
    ax.set_ylabel("perception score", fontsize=8, color=GREY)
    _style_axes(ax, ygrid=True)
    return _png(fig)


def chart_volume(periods: list, granularity: str) -> bytes:
    """Record volume per period — how much conversation each period carried."""
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    x = list(range(len(periods)))
    volumes = [p["records"] or 0 for p in periods]
    labels = [p["period"] for p in periods]
    ax.bar(x, volumes, width=0.62, color=GOLD, edgecolor=INK, linewidth=0.6)
    if x:
        peak = volumes.index(max(volumes))
        ax.annotate(f"{volumes[peak]:,}", (peak, volumes[peak]),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if granularity == "month" else 0,
                       ha="right" if granularity == "month" else "center")
    ax.set_ylabel("mentions", fontsize=8, color=GREY)
    _style_axes(ax, ygrid=True)
    return _png(fig)


# --------------------------------------------------------------------------- #
# Text safety
# --------------------------------------------------------------------------- #
# Characters outside Latin-1 that reportlab's built-in fonts CAN draw, because
# WinAnsiEncoding carries them. Keeping them means dashes and curly quotes
# survive instead of being stripped out of otherwise clean prose.
_WINANSI_EXTRA = set("–—‘’“”†‡•"
                     "…‰‹›€™ŒœŠ"
                     "šŸŽžƒˆ˜")


def _drawable(text: str) -> str:
    """Drop characters the built-in fonts would render as black boxes."""
    return "".join(ch for ch in text
                   if ord(ch) < 256 or ch in _WINANSI_EXTRA)


def _safe(text: str, limit: int = 400) -> str:
    """Make arbitrary comment text safe to typeset.

    Comments are full of emoji and non-Latin scripts. Drop what the font cannot
    draw, collapse whitespace, escape XML, and trim.
    """
    text = " ".join(str(text or "").split())
    text = _drawable(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text or "(no renderable text)"


def _rich(text: str) -> str:
    """Escape narrative prose but keep <b> and <i> working.

    The narrative may be model-written, so it is escaped like any other untrusted
    string; the two tags the writing contract permits are then restored.
    """
    text = _drawable(str(text or ""))
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for tag in ("b", "i"):
        text = (text.replace(f"&lt;{tag}&gt;", f"<{tag}>")
                    .replace(f"&lt;/{tag}&gt;", f"</{tag}>"))
    return text


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def _styles():
    base = getSampleStyleSheet()
    return {
        "country": ParagraphStyle(
            "country", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=34, leading=38, textColor=colors.white, alignment=0,
            spaceAfter=0),
        "period": ParagraphStyle(
            "period", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=13, leading=18, textColor=colors.HexColor(GOLD),
            spaceBefore=4),
        "byline": ParagraphStyle(
            "byline", parent=base["Normal"], fontSize=8.4, leading=12,
            textColor=colors.HexColor("#BFBFB4")),
        "headline": ParagraphStyle(
            "headline", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=11.5, leading=16, textColor=colors.HexColor(INK)),
        "secnum": ParagraphStyle(
            "secnum", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=17, leading=19, textColor=colors.HexColor(GOLD)),
        "sectitle": ParagraphStyle(
            "sectitle", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=13, leading=19, textColor=colors.HexColor(INK)),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=9.8, leading=15.4,
            textColor=colors.HexColor(INK_SOFT), alignment=TA_JUSTIFY,
            spaceAfter=7),
        "figtitle": ParagraphStyle(
            "figtitle", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=9.4, leading=13, textColor=colors.HexColor(INK),
            spaceBefore=6),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontSize=8.2, leading=11.5,
            textColor=colors.HexColor(GREY), spaceAfter=5),
        "notelabel": ParagraphStyle(
            "notelabel", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.2, leading=10, textColor=colors.HexColor(INK)),
        "note": ParagraphStyle(
            "note", parent=base["Normal"], fontSize=8.6, leading=12.4,
            textColor=colors.HexColor(INK_SOFT)),
        "quote": ParagraphStyle(
            "quote", parent=base["Normal"], fontSize=8.8, leading=12.8,
            textColor=colors.HexColor(INK_SOFT)),
        "meta": ParagraphStyle(
            "meta", parent=base["Normal"], fontSize=7.6, leading=10.8,
            textColor=colors.HexColor(GREY)),
        "th": ParagraphStyle(
            "th", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.6,
            leading=10, textColor=colors.HexColor(INK)),
        "td": ParagraphStyle(
            "td", parent=base["Normal"], fontSize=8.2, leading=11,
            textColor=colors.HexColor(INK_SOFT)),
    }


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _title_block(pack: dict, styles, width):
    """Near-black masthead: country, period, research-team byline."""
    country = pack["meta"]["country"]
    country = country if country and country != "All" else "All Markets"
    inner = [
        [Paragraph(_rich(country.upper()), styles["country"])],
        [Paragraph(_rich(pack["meta"]["period_label"]).upper(), styles["period"])],
        [Paragraph(BYLINE, styles["byline"])],
    ]
    band = Table(inner, colWidths=[width])
    band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(INK)),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (0, 0), 20),
        ("BOTTOMPADDING", (0, 0), (0, 0), 0),
        ("TOPPADDING", (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 1), (0, 1), 10),
        ("TOPPADDING", (0, 2), (0, 2), 8),
        ("BOTTOMPADDING", (0, 2), (0, 2), 16),
        ("LINEABOVE", (0, 2), (0, 2), 1.4, colors.HexColor(GOLD)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return band


def _headline_bar(text, styles, width):
    """The period's verdict, in a cream band under the masthead."""
    table = Table([[Paragraph(_rich(text), styles["headline"])]], colWidths=[width])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(CREAM)),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(GOLD)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return table


def _metric_strip(pack: dict, styles, width):
    """Five at-a-glance tiles: volume, the sentiment mix, net and perception."""
    sent = pack["sentiment"]
    cells = [
        ("Records in view", f"{pack['volume']['records']:,}", INK),
        ("Positive", facts.pct(sent["positive_pct"]), GREEN),
        ("Neutral", facts.pct(sent["neutral_pct"]), GREY),
        ("Negative", facts.pct(sent["negative_pct"]), RED),
        ("Perception", f"{sent['perception_score']}/100", INK),
    ]
    label_style = ParagraphStyle("ml", fontName="Helvetica", fontSize=7.0,
                                 leading=9, textColor=colors.HexColor(GREY),
                                 alignment=1)
    values, labels = [], []
    for label, value, color in cells:
        values.append(Paragraph(value, ParagraphStyle(
            "mv", fontName="Helvetica-Bold", fontSize=17, leading=20,
            textColor=colors.HexColor(color), alignment=1)))
        labels.append(Paragraph(label.upper(), label_style))
    table = Table([values, labels], colWidths=[width / len(cells)] * len(cells))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(CREAM)),
        ("LINEBELOW", (0, -1), (-1, -1), 2, colors.HexColor(GOLD)),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor(HAIRLINE)),
        ("TOPPADDING", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _section_head(number: str, title: str, styles, width):
    """'01 — MONTH IN REVIEW' over a gold rule."""
    row = [[Paragraph(number, styles["secnum"]),
            Paragraph(_rich(title).upper(), styles["sectitle"])]]
    table = Table(row, colWidths=[width * 0.09, width * 0.91])
    table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 1.6, colors.HexColor(GOLD)),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("LEFTPADDING", (1, 0), (1, 0), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
    ]))
    return table


def _data_note(text, styles, width):
    """Cream callout with a gold spine: where the evidence runs out."""
    return _callout("DATA NOTE", text, styles, width)


def _callout(label, text, styles, width):
    """The house callout: a small caps label over a cream, gold-spined block.

    Two things earn this treatment — a DATA NOTE, where the evidence runs out,
    and a HOW THIS IS MEASURED note, where a figure means something narrower
    than its name suggests. Both are the report qualifying itself, so both look
    the same on the page.
    """
    rows = [[Paragraph(_rich(label), styles["notelabel"])],
            [Paragraph(_rich(text), styles["note"])]]
    table = Table(rows, colWidths=[width])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(CREAM)),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(GOLD)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (0, 0), 7),
        ("BOTTOMPADDING", (0, 0), (0, 0), 1),
        ("TOPPADDING", (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 1), (0, 1), 8),
    ]))
    return table


def _image(png: bytes, width):
    """Scale a chart PNG to the frame width, preserving its aspect ratio."""
    img = Image(io.BytesIO(png))
    img.drawHeight = width * img.imageHeight / img.imageWidth
    img.drawWidth = width
    return img


def _figure(title, caption, png, styles, width):
    """Title + caption + chart as one unbreakable block."""
    return KeepTogether([Paragraph(_rich(title), styles["figtitle"]),
                         Paragraph(_rich(caption), styles["caption"]),
                         _image(png, width)])


def _source_link(v: dict) -> str:
    """The record's source, hyperlinked to the comment itself where we have it.

    A quote a reader cannot go and check is an assertion. The link makes every
    excerpt in this report auditable back to the platform it came from.
    """
    source = _safe(v.get("source") or "source", 40)
    url = str(v.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return source
    return f'<link href="{_safe(url, 300)}"><u>{source}</u></link>'


def _quote_card(v: dict, styles, *, label: str, theme: str = "") -> Paragraph:
    """One comment as a quote card: who said it, how loud, and where to check."""
    accent = GREEN if label == "positive" else RED
    lead = f'<b>{_rich(theme.upper())}</b> &nbsp;·&nbsp; ' if theme else ""
    meta = (f'<font color="{GREY}">{lead}</font>'
            f'<font color="{accent}"><b>{label.upper()}</b></font>'
            f'<font color="{GREY}"> &nbsp;·&nbsp; {_safe(v["author"], 40)}'
            f'{" · " + v["date"] if v["date"] else ""} &nbsp;·&nbsp; '
            f'{v["engagement"]:,} likes &nbsp;·&nbsp; '
            f'score {v["sentiment_score"]:+.2f} &nbsp;·&nbsp; '
            f'{_source_link(v)}</font>')
    return Paragraph(f'<font size="7.2">{meta}</font><br/>'
                     f'“{_safe(v["text"], 300)}”', styles["quote"])


def _card_table(rows, width):
    """Stack quote cards into one cream, hairline-ruled block."""
    table = Table(rows, colWidths=[width])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(CREAM)),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor(HAIRLINE)),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(HAIRLINE)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _theme_voice_cards(theme_voices: list, styles, width, *, themes: int = 4):
    """A real excerpt for and against each theme the analysis headlines.

    Section 03 argues theme by theme; without this it argues entirely in
    aggregates. Each card is the highest-engagement comment on that theme from
    that side — the one a visitor searching the destination actually meets.
    """
    rows = []
    for row in (theme_voices or [])[:themes]:
        for label in ("positive", "negative"):
            for v in row.get(label, [])[:1]:
                rows.append([_quote_card(v, styles, label=label,
                                         theme=row["aspect"])])
    return _card_table(rows, width) if rows else None


def _voice_cards(voices: dict, styles, width):
    """Quote cards for the most-engaged supportive and critical mentions."""
    rows = []
    for label in ("positive", "negative"):
        for v in voices.get(label, [])[:2]:
            rows.append([_quote_card(v, styles, label=label)])
    return _card_table(rows, width) if rows else None


def _peer_table(bench: dict, country: str, styles, width):
    """Peer standings: every market in the reference set, subject highlighted."""
    peers = [p for p in bench.get("peers", []) if p["reportable"]]
    if len(peers) < 2:
        return None
    header = ["Market", "Records", "Perception", "Net sentiment"]
    rows = [[Paragraph(h.upper(), styles["th"]) for h in header]]
    subject_row = None
    for i, p in enumerate(peers, start=1):
        if p["is_subject"]:
            subject_row = i
        rows.append([
            Paragraph(_rich(p["country"]), styles["td"]),
            Paragraph(f"{p['records']:,}", styles["td"]),
            Paragraph(f"{p['perception_score']}/100", styles["td"]),
            Paragraph(f"{p['net_sentiment']:+.0f}", styles["td"]),
        ])
    table = Table(rows, colWidths=[width * 0.34, width * 0.20, width * 0.23,
                                   width * 0.23])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(CREAM)),
        ("LINEBELOW", (0, 0), (-1, 0), 1.2, colors.HexColor(GOLD)),
        ("INNERGRID", (0, 1), (-1, -1), 0.5, colors.HexColor(HAIRLINE)),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(HAIRLINE)),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    if subject_row:
        style += [
            ("BACKGROUND", (0, subject_row), (-1, subject_row),
             colors.HexColor(GOLD_LIGHT)),
            ("LINEBEFORE", (0, subject_row), (0, subject_row), 3,
             colors.HexColor(GOLD)),
        ]
    table.setStyle(TableStyle(style))
    return table


def _segment_table(segments: dict, styles, width):
    """Cohort standings: share of view, perception and net per travel style."""
    cohorts = segments.get("cohorts") or []
    if not cohorts:
        return None
    header = ["Visitor segment", "Records", "Share of all records",
              "Perception", "Net"]
    rows = [[Paragraph(h.upper(), styles["th"]) for h in header]]
    faded = []
    for i, c in enumerate(cohorts, start=1):
        if not c["reportable"]:
            faded.append(i)
        rows.append([
            Paragraph(_rich(c["segment"].capitalize())
                      + ("" if c["reportable"] else "  · thin"), styles["td"]),
            Paragraph(f"{c['records']:,}", styles["td"]),
            # Whole numbers, like every other percentage in the report: this
            # column printed one decimal place while the metric strip printed
            # none, so the same kind of figure read as two kinds of precision.
            Paragraph(facts.pct(c["share_of_view_pct"]), styles["td"]),
            Paragraph(f"{c['perception_score']}/100", styles["td"]),
            Paragraph(f"{c['net_sentiment']:+.0f}", styles["td"]),
        ])
    table = Table(rows, colWidths=[width * 0.28, width * 0.16, width * 0.20,
                                   width * 0.20, width * 0.16])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(CREAM)),
        ("LINEBELOW", (0, 0), (-1, 0), 1.2, colors.HexColor(GOLD)),
        ("INNERGRID", (0, 1), (-1, -1), 0.5, colors.HexColor(HAIRLINE)),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(HAIRLINE)),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in faded:
        style.append(("TEXTCOLOR", (0, i), (-1, i), colors.HexColor(GREY)))
    table.setStyle(TableStyle(style))
    return table


def _filters_table(filters, styles, width):
    rows = [[Paragraph(f"<b>{_rich(k)}</b>", styles["meta"]),
             Paragraph(_rich(str(v)), styles["meta"])]
            for k, v in filters.items()]
    table = Table(rows, colWidths=[width * 0.30, width * 0.70])
    table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(HAIRLINE)),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor(HAIRLINE)),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _decorate(canvas, doc):
    """Gold hairline over a near-black footer strip carrying the page number."""
    canvas.saveState()
    canvas.setFillColor(colors.HexColor(INK))
    canvas.rect(0, 0, doc.pagesize[0], 12 * mm, stroke=0, fill=1)
    canvas.setStrokeColor(colors.HexColor(GOLD))
    canvas.setLineWidth(1.2)
    canvas.line(0, 12 * mm, doc.pagesize[0], 12 * mm)
    canvas.setFont("Helvetica-Bold", 7.2)
    canvas.setFillColor(colors.HexColor(GOLD))
    canvas.drawString(doc.leftMargin, 5 * mm, "AFRICA INSIGHTS")
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(colors.HexColor("#BFBFB4"))
    canvas.drawString(doc.leftMargin + 33 * mm, 5 * mm,
                      "Travel & Tourism · Perception Intelligence")
    canvas.drawRightString(doc.leftMargin + doc.width, 5 * mm,
                           f"{canvas.getPageNumber():02d}")
    canvas.restoreState()


# --------------------------------------------------------------------------- #
# Backwards-compatible narrative shim
# --------------------------------------------------------------------------- #
def narrative(df, *, country: str = "All", granularity: str = "Year",
              benchmark_df=None) -> list:
    """The report's sections as [(heading, text), ...].

    Deterministic: this is the template narrative, with no model call, so it is
    safe to use in tests and tooling that run without an API key. `build_pdf()`
    uses the full engine in `core.narrative`, which prefers Claude.
    """
    pack = facts.build(df, country=country, granularity=granularity,
                       benchmark_df=benchmark_df)
    sections, _ = narrative_engine.template_sections(pack)
    titles = narrative_engine.section_titles(pack)
    out = []
    for sid, number, _title, _brief in narrative_engine.SECTIONS:
        body = sections.get(sid)
        if body and body.get("prose"):
            out.append((f"{number} — {titles[sid]}",
                        body["prose"].replace("\n\n", " ")))
    return out


def report_title(country: str) -> str:
    where = country if country and country != "All" else "All countries"
    return f"{where} — Tourism Perception Report"


# --------------------------------------------------------------------------- #
# PDF assembly
# --------------------------------------------------------------------------- #
def build_pdf(df, *, country: str = "All", granularity: str = "Year",
              filters: dict | None = None, benchmark_df=None,
              use_claude: bool = True, return_diagnostics: bool = False):
    """Render the filtered frame as an intelligence-grade PDF.

    `df` must already have the dashboard's filters applied — the report describes
    exactly what it is given. `benchmark_df` is the same dataset with every
    filter EXCEPT country applied; supplying it enables section 05's peer
    comparison, and omitting it degrades that section to a DATA NOTE rather than
    an error. `filters` is echoed into the methodology appendix.

    Returns PDF bytes, or `(bytes, diagnostics)` when `return_diagnostics` is set
    — the diagnostics carry which engine wrote the prose and what it logged, so a
    caller can surface "written by Claude" vs "written from the template".
    """
    pack = facts.build(df, country=country, granularity=granularity,
                       benchmark_df=benchmark_df, filters=filters)

    if use_claude:
        result = narrative_engine.write(pack)
    else:
        sections, headline = narrative_engine.template_sections(pack)
        result = narrative_engine.NarrativeResult(
            sections, headline, "template",
            ["Narrative written from the built-in template (Claude disabled)."],
            narrative_engine.section_titles(pack))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, title=report_title(country),
        author="Africa INSIGHTS Research Team",
        subject="Tourism perception intelligence",
        leftMargin=17 * mm, rightMargin=17 * mm,
        topMargin=13 * mm, bottomMargin=20 * mm,
    )
    styles = _styles()
    width = doc.width
    story = [_title_block(pack, styles, width), Spacer(1, 9)]

    if result.headline:
        story += [_headline_bar(result.headline, styles, width), Spacer(1, 9)]

    if not pack["volume"]["records"]:
        story.append(Paragraph(
            "No records match the current filters, so there is nothing to "
            "report. Widen the year range, sentiment or data-source filters and "
            "generate the report again.", styles["body"]))
        doc.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
        return (buf.getvalue(), _diagnostics(pack, result)) if return_diagnostics \
            else buf.getvalue()

    story += [_metric_strip(pack, styles, width)]
    # The tiles are the most-read figures in the report and the only ones with
    # no room to carry their own denominator, so it goes underneath them.
    story += [Paragraph(
        f"Positive, neutral and negative are shares of all "
        f"{pack['volume']['records']:,} records in view. Perception is a 0-100 "
        f"score where 50 is neutral.", styles["caption"]), Spacer(1, 4)]

    # Charts and tables are attached to the section that argues from them, so a
    # reader never has to hold a number in their head across a page turn.
    granularity_key = pack["trend"]["granularity"]
    attachments = {
        "perception_overview": lambda: [
            _figure("Figure 1 — Sentiment breakdown",
                    f"How the {pack['volume']['records']:,} records in view "
                    f"split across sentiment classes; each share is a percentage "
                    f"of all records in view. Net sentiment is "
                    f"{pack['sentiment']['net_sentiment']:+.0f} points.",
                    chart_sentiment_breakdown(pack["sentiment"]), styles, width)],
        "thematic_analysis": _thematic_attachments(pack, styles, width),
        "visitor_segments": _segment_attachments(pack, styles, width),
        "competitive_benchmarking": lambda: _optional(
            _peer_table(pack["benchmark"], country, styles, width)),
        "trends_signals": _trend_attachments(pack, granularity_key, styles, width),
    }

    for sid, number, title, prose, notes in result.ordered():
        head = [_section_head(number, title, styles, width), Spacer(1, 6)]
        paragraphs = [Paragraph(_rich(p), styles["body"])
                      for p in prose.split("\n\n") if p.strip()]
        # Keep the heading with its first paragraph so a section title can never
        # be stranded alone at the foot of a page.
        if paragraphs:
            story.append(KeepTogether(head + paragraphs[:1]))
            story.extend(paragraphs[1:])
        else:
            story.extend(head)
        for note in notes:
            story += [Spacer(1, 2), _data_note(note, styles, width), Spacer(1, 6)]
        maker = attachments.get(sid)
        if maker:
            for element in maker():
                story += [Spacer(1, 4), element]
        story.append(Spacer(1, 12))

    # --- Methodology ---------------------------------------------------------
    story.append(_section_head("08", "Methodology & Provenance", styles, width))
    story.append(Spacer(1, 8))
    active = dict(filters or {})
    active.setdefault("Country", country)
    active.setdefault("Period", pack["meta"]["period_label"])
    active.setdefault("Records", f"{pack['volume']['records']:,}")
    active["Generated"] = pack["meta"]["generated_utc"]
    active["Narrative"] = _engine_label(result.engine)
    story.append(_filters_table(active, styles, width))
    story.append(Spacer(1, 8))
    prov = pack["provenance"]
    lines = []
    if prov["sources"]:
        lines.append("Evidence base: " + ", ".join(
            facts.source_phrase(s) for s in prov["sources"]) + ".")
    if prov["content_attributed_pct"] is not None:
        lines.append(f"{facts.pct(prov['content_attributed_pct'])} of all "
                     f"records in view were attributed to this market from what "
                     f"the text itself is about rather than from the query that "
                     f"surfaced them.")
    lines.append(
        "Percentages are written as whole numbers throughout, and each is stated "
        "with what it is a share of: sentiment shares are of all records in "
        "view, a theme's shares are of that theme's own mentions, and a cohort's "
        "shares are of the records in that cohort. Shares on different "
        "denominators are not comparable with one another.")
    if pack["segments"].get("method"):
        lines.append(pack["segments"]["method"] + " "
                     + pack["segments"]["limitations"])
    lines.append(
        f"Sentiment is scored by a multilingual transformer; themes, emotion and "
        f"visitor segment are rule-based taggers. Evidence thresholds: "
        f"{MIN_SAMPLE} records for a reliable view, {MIN_ASPECT_MENTIONS} "
        f"mentions before a theme can headline a finding, {NET_MARGIN:.0f} points "
        f"before a theme counts as praised or criticised, "
        f"{rails.GAP_MARGIN:.0f} points before a cross-country gap is stated.")
    lines.append(
        "Every figure in this report is computed from the filtered records by "
        "the analysis pipeline. The written analysis interprets those figures; "
        "each number it states is verified back against them before publication, "
        "and any section that fails verification is replaced by the deterministic "
        "summary.")
    story.append(Paragraph(_rich(" ".join(lines)), styles["meta"]))

    doc.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    pdf = buf.getvalue()
    return (pdf, _diagnostics(pack, result)) if return_diagnostics else pdf


def _optional(element):
    return [element] if element is not None else []


def _segment_attachments(pack, styles, width):
    """The cohort table, then the note on how the cohorts were inferred.

    The note is not an appendix footnote: it sits directly under the figures it
    qualifies, because a reader who takes "luxury travellers" for a surveyed
    population will over-read every number in this section.
    """
    def make():
        segments = pack["segments"]
        out = []
        table = _segment_table(segments, styles, width)
        if table is not None:
            out.append(KeepTogether([
                Paragraph("Table 1 — Visitor segments", styles["figtitle"]),
                Paragraph("Share of all records is that cohort's share of every "
                          "record in view; perception and net sentiment are "
                          "computed within the cohort itself. Cohorts below the "
                          f"{MIN_SAMPLE}-record floor are faded and marked thin.",
                          styles["caption"]),
                table]))
        if segments.get("method"):
            out += [Spacer(1, 4),
                    _callout("HOW VISITOR SEGMENTS ARE INFERRED",
                             segments["method"] + " " + segments["limitations"],
                             styles, width)]
        return out
    return make


def _thematic_attachments(pack, styles, width):
    def make():
        out = []
        # Voices first: they follow directly from the prose that quotes them, and
        # being short they fill the tail of the page instead of leaving a gap
        # that the full-height theme chart could never fit into.
        theme_cards = _theme_voice_cards(pack.get("theme_voices"), styles, width)
        if theme_cards is not None:
            out.append(KeepTogether([
                Paragraph("Voices behind the numbers — by theme",
                          styles["figtitle"]),
                Paragraph("For each headline theme, the highest-engagement "
                          "supportive and critical comment that raised it, with "
                          "its sentiment and a link to the source. Emoji and "
                          "non-Latin characters are stripped for print.",
                          styles["caption"]),
                theme_cards]))
        cards = _voice_cards(pack["voices"], styles, width)
        if cards is not None:
            out.append(KeepTogether([
                Paragraph("The loudest voices overall", styles["figtitle"]),
                Paragraph("The most-engaged supportive and critical mentions "
                          "anywhere in this view, whatever theme they raise.",
                          styles["caption"]),
                cards]))
        if pack["aspects"]:
            out.append(_figure(
                "Figure 2 — Sentiment by theme",
                "Every travel theme mentioned in this view, ranked by net "
                "sentiment. Each bar is a percentage of that theme's own "
                "mentions, not of the whole view, so bars are not comparable "
                "in volume. Mention counts in brackets; themes below the "
                f"{MIN_ASPECT_MENTIONS}-mention floor are faded and marked thin.",
                chart_aspect_sentiment(pack["aspects"]), styles, width))
        return out
    return make


def _trend_attachments(pack, granularity, styles, width):
    def make():
        periods = pack["trend"]["periods"]
        if not periods:
            return []
        return [
            _figure(f"Figure 3 — Perception over time (by {granularity})",
                    "Perception score per period on a 0–100 scale, where 50 "
                    "is neutral.",
                    chart_trend(periods, granularity), styles, width),
            _figure(f"Figure 4 — Conversation volume (by {granularity})",
                    "Records per period. Read the trend above against this: a "
                    "swing on thin volume is noise.",
                    chart_volume(periods, granularity), styles, width),
        ]
    return make


def _engine_label(engine: str) -> str:
    return {
        "claude": f"Written by {narrative_engine.MODEL}; every figure verified "
                  f"against the computed data.",
        "claude+template": f"Written by {narrative_engine.MODEL}; sections that "
                           f"failed figure verification were replaced by the "
                           f"deterministic summary.",
        "template": "Deterministic summary (no language model was available).",
    }.get(engine, engine)


def _diagnostics(pack: dict, result) -> dict:
    return {
        "engine": result.engine,
        "log": list(result.log),
        "headline": result.headline,
        "sections": [sid for sid, *_ in narrative_engine.SECTIONS
                     if result.sections.get(sid, {}).get("prose")],
        "data_notes": pack["data_notes"],
        "records": pack["volume"]["records"],
    }
