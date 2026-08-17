"""The fact pack — every number the report is allowed to print.

`build(df, ...)` takes an ALREADY-FILTERED frame and returns a plain, JSON-safe
dict of computed figures: the sentiment mix, per-aspect net scores, the trend,
visitor segments, peer benchmarking, top voices and source provenance.

This module is the report's single source of numerical truth. Nothing downstream
— not the charts, not the deterministic template, and above all not the language
model that writes the narrative — computes a figure of its own. The model is
handed this dict and writes prose *around* it; `allowed_numbers()` then lets the
validator prove that every number in the finished prose came from here.

Two consequences worth keeping in mind when editing:

  * Every figure the narrative may need must appear here. A number the model can
    reason about but cannot find in the pack will be rejected by the validator,
    and the section will silently fall back to the template.
  * Everything must stay JSON-serializable (the pack is sent over the wire), so
    numpy scalars are cast on the way out. `_num()` and `_int()` do that.

Deliberately free of matplotlib, reportlab, Streamlit and the Anthropic SDK, so
it can be imported and tested on its own against data/records.csv.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from core import rails

SENTIMENTS = ("positive", "neutral", "negative")

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

UNCLASSIFIED = "unclassified"


# --------------------------------------------------------------------------- #
# Scalar helpers — keep the pack JSON-safe and free of numpy types
# --------------------------------------------------------------------------- #
def _num(value, digits: int = 1):
    """Round to `digits` and return a plain float (never a numpy scalar)."""
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


def _int(value):
    return None if value is None else int(value)


def _month_name(ts) -> str:
    return ts.strftime("%B %Y") if ts is not None else "n/a"


def date_span(df):
    """(earliest, latest) timestamps in the frame, or (None, None)."""
    if df.empty or "timestamp" not in df.columns:
        return None, None
    stamps = df["timestamp"].dropna()
    if stamps.empty:
        return None, None
    return stamps.min(), stamps.max()


def period_label(lo, hi) -> str:
    """'March 2026' for a single month, 'March 2025 – March 2026' for a span.

    The house style puts a single month under the country name. Reports are
    generated from whatever the dashboard is filtered to, which is usually wider
    than a month, so the label widens with the data rather than claiming a month
    the records do not cover.
    """
    if lo is None or hi is None:
        return "No dated records"
    if (lo.year, lo.month) == (hi.year, hi.month):
        return _month_name(lo)
    return f"{_month_name(lo)} – {_month_name(hi)}"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def summarize(df) -> dict:
    """Headline metrics for a frame: counts, percentages, net, perception."""
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
    """Perception score, volume and negative share per period."""
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
    """Split the frame into earlier and later halves of its own date range."""
    if df.empty or "timestamp" not in df.columns:
        return df.iloc[0:0], df.iloc[0:0]
    ordered = df.sort_values("timestamp")
    midpoint = len(ordered) // 2
    if midpoint == 0:
        return ordered.iloc[0:0], ordered
    return ordered.iloc[:midpoint], ordered.iloc[midpoint:]


# --------------------------------------------------------------------------- #
# Fact-pack sections
# --------------------------------------------------------------------------- #
def _sentiment_facts(metrics: dict) -> dict:
    return {
        "positive_records": _int(metrics["split"]["positive"]),
        "neutral_records": _int(metrics["split"]["neutral"]),
        "negative_records": _int(metrics["split"]["negative"]),
        "positive_pct": _num(metrics["pos"]),
        "neutral_pct": _num(metrics["neu"]),
        "negative_pct": _num(metrics["neg"]),
        "net_sentiment": _num(metrics["net"]),
        "perception_score": _int(metrics["perception"]),
        "perception_scale": "0-100, where 50 is neutral",
    }


def _aspect_facts(table: pd.DataFrame) -> list:
    out = []
    for _, row in table.iterrows():
        mentions = int(row["total"])
        out.append({
            "aspect": row["aspect"],
            "mentions": mentions,
            "positive_pct": _num(row["positive"]),
            "neutral_pct": _num(row["neutral"]),
            "negative_pct": _num(row["negative"]),
            "net": _num(row["net"]),
            # Below the mention floor a theme may be described but must not
            # headline a finding — the narrative prompt enforces this.
            "reportable": rails.has_enough(mentions),
        })
    return out


def _aspect_movement(df) -> list:
    """Change in each theme's negative share between the two halves of the window."""
    earlier, later = split_halves(df)
    if (len(earlier) < rails.MIN_ASPECT_MENTIONS
            or len(later) < rails.MIN_ASPECT_MENTIONS):
        return []
    before = aspect_table(earlier).set_index("aspect")
    after = aspect_table(later).set_index("aspect")
    out = []
    for aspect in after.index:
        if aspect not in before.index:
            continue
        mentions_after = int(after.loc[aspect, "total"])
        mentions_before = int(before.loc[aspect, "total"])
        change = float(after.loc[aspect, "negative"] - before.loc[aspect, "negative"])
        direction = ("worsening" if change >= rails.NET_MARGIN else
                     "improving" if change <= -rails.NET_MARGIN else "stable")
        out.append({
            "aspect": aspect,
            "mentions_first_half": mentions_before,
            "mentions_second_half": mentions_after,
            "negative_pct_first_half": _num(before.loc[aspect, "negative"]),
            "negative_pct_second_half": _num(after.loc[aspect, "negative"]),
            "change_points": _num(change),
            "direction": direction,
            "reportable": (rails.has_enough(mentions_before)
                           and rails.has_enough(mentions_after)),
        })
    return sorted(out, key=lambda r: -(r["change_points"] or 0))


def _segment_facts(df, n: int) -> dict:
    if "segment" not in df.columns or not n:
        return {"available": False, "note": "No segment column in this dataset.",
                "classified_pct": None, "cohorts": []}
    cohorts = df.loc[df["segment"].ne(UNCLASSIFIED) & df["segment"].ne(""), "segment"]
    if cohorts.empty:
        return {"available": False,
                "note": "No record in this view carries a travel-style signal.",
                "classified_pct": 0.0, "cohorts": []}
    rows = []
    for segment, count in cohorts.value_counts().items():
        slice_ = df[df["segment"] == segment]
        stats = summarize(slice_)
        rows.append({
            "segment": segment,
            "records": _int(count),
            "share_of_view_pct": _num(100 * count / n),
            "perception_score": _int(stats["perception"]),
            "net_sentiment": _num(stats["net"]),
            "positive_pct": _num(stats["pos"]),
            "negative_pct": _num(stats["neg"]),
            "reportable": not rails.is_thin(count),
        })
    comparable = [r for r in rows if r["reportable"]]
    return {
        "available": True,
        "note": "",
        "classified_pct": _num(100 * len(cohorts) / n),
        "unclassified_records": _int(n - len(cohorts)),
        "cohorts": rows,
        "comparable_cohorts": len(comparable),
    }


def _benchmark_facts(benchmark_df, country: str, metrics: dict,
                     table: pd.DataFrame) -> dict:
    """Peer comparison against the other countries in the reference frame.

    `benchmark_df` is the same dataset with every filter EXCEPT country applied,
    so peers are compared on like-for-like years, sources and relevance.
    """
    unavailable = {
        "available": False,
        "note": ("No cross-country reference frame was supplied, so competitive "
                 "benchmarking could not be computed for this report."),
        "peers": [], "aspect_gaps": [], "rank": None,
    }
    if benchmark_df is None or not len(benchmark_df):
        return unavailable
    if not country or country == "All":
        return {**unavailable,
                "note": ("This report covers all countries at once, so there is "
                         "no single subject to benchmark against its peers.")}
    if "country" not in benchmark_df.columns:
        return unavailable

    peers = []
    for name, slice_ in benchmark_df.groupby("country"):
        stats = summarize(slice_)
        peers.append({
            "country": name,
            "records": _int(stats["n"]),
            "perception_score": _int(stats["perception"]),
            "net_sentiment": _num(stats["net"]),
            "positive_pct": _num(stats["pos"]),
            "negative_pct": _num(stats["neg"]),
            "is_subject": name == country,
            "reportable": not rails.is_thin(stats["n"]),
        })
    peers.sort(key=lambda r: -(r["perception_score"] or 0))
    ranked = [r for r in peers if r["reportable"]]
    if len(ranked) < 2:
        return {**unavailable,
                "note": (f"Only {len(ranked)} country in the reference frame clears "
                         f"the {rails.MIN_SAMPLE}-record threshold, so there is no "
                         f"peer group to benchmark against.")}

    position = next((i + 1 for i, r in enumerate(ranked) if r["is_subject"]), None)
    rank = ({"position": position, "of": len(ranked)} if position else None)

    # The distance to the top of the table is the number a tourism board asks
    # for first, so compute it here rather than leaving the writer to subtract
    # two scores itself — a derived figure the narrative may not invent.
    leader = ranked[0]
    if rank:
        rank["leader"] = leader["country"]
        rank["leader_perception"] = leader["perception_score"]
        rank["points_behind_leader"] = _num(
            leader["perception_score"] - metrics["perception"], 0)

    # Per-aspect gap: the subject's net score against the median of its peers,
    # stated only where both sides clear the mention floor and the gap clears
    # GAP_MARGIN (a gap carries the error of both numbers, so it needs a wider
    # margin than a single net score does).
    gaps = []
    subject_nets = {r["aspect"]: r for r in _aspect_facts(table) if r["reportable"]}
    peer_frame = benchmark_df[benchmark_df["country"] != country]
    peer_table = aspect_table(peer_frame)
    peer_nets = {r["aspect"]: r for r in _aspect_facts(peer_table) if r["reportable"]}
    for aspect, subject in subject_nets.items():
        peer = peer_nets.get(aspect)
        if peer is None:
            continue
        gap = subject["net"] - peer["net"]
        gaps.append({
            "aspect": aspect,
            "country_net": subject["net"],
            "country_mentions": subject["mentions"],
            "peer_net": peer["net"],
            "peer_mentions": peer["mentions"],
            "gap_points": _num(gap),
            "direction": "ahead" if gap > 0 else "behind",
            "reportable": abs(gap) >= rails.GAP_MARGIN,
        })
    gaps.sort(key=lambda r: -abs(r["gap_points"] or 0))
    return {
        "available": True,
        "note": "",
        "peer_countries": len(ranked) - (1 if position else 0),
        "peers": peers,
        "rank": rank,
        "aspect_gaps": gaps,
        "gap_margin_points": rails.GAP_MARGIN,
    }


def _trend_facts(series: pd.DataFrame, granularity: str, thin: bool) -> dict:
    periods = [{
        "period": row["period"],
        "perception_score": _num(row["perception"], 0),
        "records": _int(row["volume"]),
        "negative_pct": _num(row["neg_pct"]),
    } for _, row in series.iterrows()]
    out = {
        "granularity": granularity.lower(),
        "period_count": len(periods),
        "periods": periods,
        "readable": bool(len(periods) >= 2 and not thin),
    }
    if len(periods) >= 2:
        first, last = periods[0], periods[-1]
        change = (last["perception_score"] or 0) - (first["perception_score"] or 0)
        out.update({
            "first_period": first["period"],
            "first_perception": first["perception_score"],
            "last_period": last["period"],
            "last_perception": last["perception_score"],
            "change_points": _num(change, 0),
            "direction": ("improving" if change >= rails.TREND_MARGIN else
                          "declining" if change <= -rails.TREND_MARGIN else "flat"),
        })
        peak = max(periods, key=lambda p: p["records"] or 0)
        out["peak_period"] = peak["period"]
        out["peak_records"] = peak["records"]
    return out


def _voice_facts(df, limit: int = 3) -> dict:
    """Most-engaged supportive and critical mentions, as quotable evidence."""
    out = {"positive": [], "negative": []}
    if df.empty:
        return out
    for label in ("positive", "negative"):
        rows = (df[df["sentiment_label"] == label]
                .sort_values("engagement", ascending=False).head(limit))
        for _, row in rows.iterrows():
            when = ""
            if pd.notna(row.get("timestamp")):
                when = pd.Timestamp(row["timestamp"]).strftime("%b %Y")
            out[label].append({
                "text": str(row.get("text") or "")[:400],
                "author": str(row.get("author") or "anonymous"),
                "date": when,
                "engagement": _int(row.get("engagement") or 0),
                "sentiment_score": _num(row.get("sentiment_score") or 0, 2),
                "aspects": str(row.get("aspects") or ""),
                "source": SOURCE_LABELS.get(row.get("source"), row.get("source")),
                "url": str(row.get("url") or ""),
            })
    return out


def source_phrase(source: dict) -> str:
    """'1,185 YouTube comments' / '1 YouTube transcript segment'.

    The labels are stored plural because that is the usual case; a single record
    still has to read as English.
    """
    label = source["source"]
    if source["records"] == 1 and label.endswith("s"):
        label = label[:-1]
    return f"{source['records']:,} {label}"


def _provenance_facts(df, n: int, country: str) -> dict:
    counts = df["source"].value_counts()
    sources = [{"source": SOURCE_LABELS.get(s, s), "records": _int(c),
                "share_pct": _num(100 * c / n)} for s, c in counts.items()]
    out = {"sources": sources, "content_attributed_pct": None, "emotions": []}
    if "country_source" in df.columns and country and country != "All":
        content = int(df["country_source"].isin(["content", "context"]).sum())
        out["content_attributed_pct"] = _num(100 * content / n)
    emotive = df.loc[df["emotion"].ne("neutral") & df["emotion"].ne(""), "emotion"]
    if not emotive.empty:
        out["emotions"] = [{"emotion": e, "records": _int(c),
                            "share_pct": _num(100 * c / n)}
                           for e, c in emotive.value_counts().head(4).items()]
    return out


def _data_notes(volume, aspects, segments, benchmark, trend) -> list:
    """Every place the evidence is too thin to carry a claim.

    Computed here rather than left to the writer's judgement, so the same gap
    always produces the same caveat whether Claude or the template wrote the
    prose. The narrative prompt requires each of these to surface as a DATA NOTE.
    """
    notes = []
    n = volume["records"]
    if volume["is_thin"]:
        notes.append(
            f"This view holds {n:,} records, below the {rails.MIN_SAMPLE}-record "
            f"threshold for a reliable read. Every figure in this report is "
            f"indicative only and should not be used to set direction on its own.")
    reportable = [a for a in aspects if a["reportable"]]
    if aspects and not reportable:
        notes.append(
            f"No theme reaches the {rails.MIN_ASPECT_MENTIONS}-mention floor "
            f"required to headline a finding, so the thematic analysis describes "
            f"the mix without ranking it.")
    elif len(reportable) < len(aspects):
        thin = [a["aspect"] for a in aspects if not a["reportable"]]
        notes.append(
            f"{len(thin)} theme{'' if len(thin) == 1 else 's'} fall"
            f"{'s' if len(thin) == 1 else ''} below the "
            f"{rails.MIN_ASPECT_MENTIONS}-mention floor and "
            f"{'is' if len(thin) == 1 else 'are'} excluded from the headline "
            f"findings: {', '.join(thin)}.")
    if not segments.get("available"):
        notes.append(segments.get("note") or "Visitor segments are unavailable.")
    elif segments.get("comparable_cohorts", 0) < 2:
        comparable = segments.get("comparable_cohorts", 0)
        notes.append(
            (f"No visitor cohort clears" if not comparable else
             f"Only one visitor cohort clears")
            + f" the {rails.MIN_SAMPLE}-record threshold, so cohorts cannot be "
              f"compared against each other in this view.")
    if not benchmark.get("available"):
        notes.append(benchmark["note"])
    elif not any(g["reportable"] for g in benchmark.get("aspect_gaps", [])):
        notes.append(
            f"No theme shows a cross-country gap wider than the "
            f"{rails.GAP_MARGIN:.0f}-point margin this report requires before "
            f"stating one, so benchmarking is limited to overall scores.")
    if not trend["readable"]:
        notes.append(
            "There is not enough volume spread across periods to read a "
            "direction of travel; the trend is shown for completeness only.")
    return notes


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build(df, *, country: str = "All", granularity: str = "Year",
          benchmark_df=None, filters: dict | None = None) -> dict:
    """Compute the full fact pack for an already-filtered frame."""
    metrics = summarize(df)
    n = metrics["n"]
    lo, hi = date_span(df)
    table = aspect_table(df)
    series = time_series(df, granularity)
    thin = rails.is_thin(n)

    volume = {
        "records": _int(n),
        "is_thin": bool(thin),
        "min_sample": rails.MIN_SAMPLE,
    }
    aspects = _aspect_facts(table)
    segments = _segment_facts(df, n)
    benchmark = _benchmark_facts(benchmark_df, country, metrics, table)
    trend = _trend_facts(series, granularity, thin)

    pack = {
        "meta": {
            "country": country,
            "period_label": period_label(lo, hi),
            "period_start": _month_name(lo),
            "period_end": _month_name(hi),
            "granularity": granularity.lower(),
            "generated_utc": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
            "filters": dict(filters or {}),
            "thresholds": {
                "min_sample_records": rails.MIN_SAMPLE,
                "min_aspect_mentions": rails.MIN_ASPECT_MENTIONS,
                "net_margin_points": rails.NET_MARGIN,
                "gap_margin_points": rails.GAP_MARGIN,
                "trend_margin_points": rails.TREND_MARGIN,
            },
        },
        "volume": volume,
        "sentiment": _sentiment_facts(metrics),
        "aspects": aspects,
        "aspect_movement": _aspect_movement(df) if not thin else [],
        "segments": segments,
        "benchmark": benchmark,
        "trend": trend,
        "voices": _voice_facts(df),
        "provenance": _provenance_facts(df, n, country) if n else
                      {"sources": [], "content_attributed_pct": None, "emotions": []},
    }
    pack["data_notes"] = _data_notes(volume, aspects, segments, benchmark, trend)
    return pack


# --------------------------------------------------------------------------- #
# Number allowlist — what the narrative validator checks prose against
# --------------------------------------------------------------------------- #
def _walk_numbers(node, out: set) -> None:
    if isinstance(node, bool):
        return                                  # bools are ints in Python; skip
    if isinstance(node, (int, float)):
        if not (isinstance(node, float) and (math.isnan(node) or math.isinf(node))):
            out.add(float(node))
        return
    if isinstance(node, dict):
        for key, value in node.items():
            # Author handles and URLs are identifiers: "@johnajah4752" would
            # otherwise licence 4752 as a publishable statistic. Quoted comment
            # `text` IS harvested — a number the writer reproduces from a quote
            # we supplied is repetition, not invention, and the quote sits on the
            # same page for the reader to check.
            if key in ("url", "author"):
                continue
            _walk_numbers(value, out)
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            _walk_numbers(item, out)
        return
    if isinstance(node, str):
        # Period labels ("2019", "2026-03") and dates carry numbers the prose
        # legitimately cites, so harvest them too.
        buffer = ""
        for char in node + " ":
            if char.isdigit():
                buffer += char
            else:
                if buffer:
                    out.add(float(buffer))
                buffer = ""


def allowed_numbers(pack: dict) -> set:
    """Every numeric value the narrative is permitted to state.

    Includes each figure as computed plus its rounded renderings, because prose
    says "62%" where the pack holds 61.7, and "3,042" where it holds 3042.
    """
    raw: set = set()
    _walk_numbers(pack, raw)
    allowed: set = set()
    for value in raw:
        allowed.add(value)
        allowed.add(abs(value))
        for digits in (0, 1):
            allowed.add(round(value, digits))
            allowed.add(abs(round(value, digits)))
    # Small integers cover ordinary enumeration ("three themes", "the top 5"),
    # which the writer derives from the pack's own lists rather than inventing.
    allowed.update(float(i) for i in range(0, 21))
    return allowed
