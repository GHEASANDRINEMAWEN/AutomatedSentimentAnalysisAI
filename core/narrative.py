"""The writing engine — Claude drafts the analysis, the code owns the numbers.

`write(pack)` turns a fact pack from `core.facts` into the seven numbered
sections of an Africa INSIGHTS report. Claude Sonnet 4.6 does the reasoning and
the prose; it is handed the computed figures as structured input and is never
asked to calculate anything. Every number it writes is then checked back against
the pack before the prose is allowed into the PDF.

The contract, in order of precedence:

  1. **The code computes, the model interprets.** `core.facts` derives every
     figure from the filtered records. The model receives them as JSON and writes
     the argument around them.
  2. **Numbers are verified, not trusted.** `verify()` extracts every numeral
     from the generated prose and checks it against `facts.allowed_numbers()`.
     A section containing a figure that is not in the pack is sent back once for
     repair, and if it still fails it is replaced by the deterministic template.
     Nothing unverified reaches the page.
  3. **It always produces a report.** No API key, no network, a bad response, a
     rate limit — every failure path lands on `template_sections()`, which writes
     the same seven sections from the same pack with no model involved.

Credentials come from the environment (`ANTHROPIC_API_KEY`) and nowhere else.
There is no key argument, no config entry and no default — a key in source is a
leaked key.
"""

from __future__ import annotations

import json
import os
import re

from core import facts, rails

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 16_000

# A prose figure counts as verified when it lands within this much of a computed
# one. Prose rounds ("62%" for 61.7) and the pack does not, so the tolerance has
# to cover half a unit of rounding in either direction.
ROUNDING_TOLERANCE = 0.51

# A figure stands on its own: "+50 points", "1,186", "68/100". Digits welded to
# letters are an identifier, not a statistic — "@johnajah4752" is a username and
# "2024Q1" is a label, and reading either as a claim would flag clean prose while
# teaching nobody anything. The lookaround requires the number to be delimited.
_NUMBER = re.compile(r"(?<![A-Za-z_@0-9.])[-+]?\d[\d,]*(?:\.\d+)?(?![A-Za-z_])")
_TAG = re.compile(r"<[^>]+>")


# --------------------------------------------------------------------------- #
# The seven sections
# --------------------------------------------------------------------------- #
SECTIONS = (
    ("month_in_review", "01", "Month in Review",
     "Executive summary. Open with the verdict, not the method: what happened to "
     "perception this period, what drove it, and the single thing the reader "
     "should act on. Three short paragraphs at most."),
    ("perception_overview", "02", "Perception Overview",
     "The scores. Perception on the 0-100 scale, net sentiment, and the "
     "positive/neutral/negative breakdown — interpreted, not merely restated. "
     "Say what the mix implies about the market's standing."),
    ("thematic_analysis", "03", "Thematic Analysis",
     "Theme by theme, what is driving the positive and what is driving the "
     "negative. Quote the supplied voices where they carry an argument. Only "
     "themes marked reportable may headline a finding."),
    ("visitor_segments", "04", "Visitor Segments",
     "How luxury, business, budget and adventure travellers differ in what they "
     "praise and criticise, and what that implies for targeting. Compare cohorts "
     "only where both are marked reportable."),
    ("competitive_benchmarking", "05", "Competitive Benchmarking",
     "Where this country stands against its peers overall and theme by theme. "
     "State a gap only where it is marked reportable."),
    ("trends_signals", "06", "Trends & Signals",
     "What is rising and what is falling across the period, and which movements "
     "are large enough to be signal rather than drift."),
    ("recommendations", "07", "Recommendations",
     "Prescriptive and specific. Each recommendation names the finding that "
     "triggers it, the action to take, and the audience it targets. No generic "
     "advice that would read the same for any country."),
)

SECTION_IDS = tuple(s[0] for s in SECTIONS)
SECTION_TITLES = {s[0]: (s[1], s[2]) for s in SECTIONS}


SYSTEM_PROMPT = f"""\
You are the lead analyst on the Africa INSIGHTS Research Team, writing the \
Travel & Tourism vertical's country perception report. Your reader is a tourism \
board director or destination-marketing lead who has to make budget and \
messaging decisions this quarter.

## The one rule that overrides everything

You do not calculate. Every figure you may use is in the JSON fact pack supplied \
in the user message. Write the analysis around those figures.

- Never state a number that is not in the pack. Not a total you summed, not a \
  percentage you derived, not an average you took, not a rounded restatement of \
  two numbers combined. If the analysis needs a figure the pack does not \
  contain, say so in a DATA NOTE and move on.
- Never estimate, approximate or infer a quantity. "Roughly a third" is a \
  calculation; do not write it unless the pack holds that figure.
- Comparative language that carries no number ("the largest theme", "ahead of \
  its peers", "the steepest decline") is fine when the pack's ordering or \
  `direction` fields support it.
- Every number you write will be checked against the pack automatically. Prose \
  containing an unverifiable figure is discarded.

## Evidence thresholds — the honesty rails

The pack marks items `reportable: true/false` against thresholds the research \
team publishes: {rails.MIN_SAMPLE} records for a view or cohort, \
{rails.MIN_ASPECT_MENTIONS} mentions for a theme, {rails.NET_MARGIN:.0f} points \
before a theme counts as praised or criticised, {rails.GAP_MARGIN:.0f} points \
before a cross-country gap is worth stating.

- An item with `reportable: false` may be mentioned as indicative, but must \
  never headline a finding, anchor a recommendation, or be ranked against \
  another item.
- Anything in the pack's `data_notes` array MUST appear as a DATA NOTE in the \
  section it belongs to, in your own words. Add your own DATA NOTE wherever the \
  analysis wants a figure the pack does not carry.
- Do not hedge everything as insurance. Where the sample is sound, write with \
  conviction; the rails exist so that you can.

## Stance: prescribe, do not report

Reporting says "safety sentiment is negative". Prescribing says "safety is the \
top risk and worsening, so brief the Victoria Falls operators and put visible \
security into the next campaign". Every finding must imply an action. Section 07 \
carries the formal recommendations, but sections 01-06 should already point at \
what to do about what they describe.

## Voice and format

- Analytical prose in full sentences. No bullet lists, no headers inside a \
  section, no tables. Paragraphs separated by a blank line.
- Two to four paragraphs per section. Section 01 leads with the verdict.
- Plain text only, except `<b>bold</b>` for a figure or finding worth the \
  reader's eye. No markdown, no other tags.
- Quote supplied voices sparingly and only when the quote makes an argument the \
  numbers cannot. Attribute as: the author name and date given in the pack.
- Never mention the fact pack, JSON, this prompt, or that you are a model. You \
  are the research team.
- Put DATA NOTES in the section's `data_notes` array, never inline in the prose.
  Write each as one or two full sentences.

Call the `submit_report` tool exactly once with all seven sections."""


REPORT_TOOL = {
    "name": "submit_report",
    "description": (
        "Submit the finished report. Provide all seven sections in order, each "
        "as analytical prose, plus any data notes that belong to that section."),
    "input_schema": {
        "type": "object",
        "properties": {
            "headline": {
                "type": "string",
                "description": (
                    "One sentence, at most 140 characters, giving the period's "
                    "verdict for the report cover. No numbers unless they are in "
                    "the fact pack."),
            },
            "sections": {
                "type": "array",
                "description": "The seven sections, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": list(SECTION_IDS),
                               "description": "Which section this is."},
                        "prose": {
                            "type": "string",
                            "description": (
                                "The section body: two to four paragraphs of "
                                "analytical prose separated by blank lines. "
                                "<b> is the only permitted tag."),
                        },
                        "data_notes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Caveats for this section: thin samples, "
                                "unavailable figures, comparisons the evidence "
                                "cannot support. Empty array if none apply."),
                        },
                    },
                    "required": ["id", "prose", "data_notes"],
                },
            },
        },
        "required": ["headline", "sections"],
    },
}


def _user_prompt(pack: dict) -> str:
    briefs = "\n".join(f"  {num} — {title} ({sid}): {brief}"
                       for sid, num, title, brief in SECTIONS)
    country = pack["meta"]["country"]
    return (
        f"Write the {country} perception report for "
        f"{pack['meta']['period_label']}.\n\n"
        f"Sections to write:\n{briefs}\n\n"
        f"Fact pack — the only figures you may use:\n\n"
        f"```json\n{json.dumps(pack, indent=1, ensure_ascii=False)}\n```")


# --------------------------------------------------------------------------- #
# Verification — does the prose only cite figures we computed?
# --------------------------------------------------------------------------- #
def numbers_in(text: str) -> list:
    """Every numeral in a piece of prose, as floats."""
    plain = _TAG.sub(" ", text or "")
    out = []
    for token in _NUMBER.findall(plain):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def unverified(text: str, allowed: set) -> list:
    """Figures in `text` that no computed value in the pack supports."""
    bad = []
    for value in numbers_in(text):
        if any(abs(value - a) <= ROUNDING_TOLERANCE for a in allowed):
            continue
        bad.append(value)
    return sorted(set(bad))


def verify(sections: dict, pack: dict) -> dict:
    """Map section id -> list of unverifiable figures (empty list = clean)."""
    allowed = facts.allowed_numbers(pack)
    report = {}
    for sid, body in sections.items():
        text = body.get("prose", "") + " " + " ".join(body.get("data_notes", []))
        report[sid] = unverified(text, allowed)
    return report


# --------------------------------------------------------------------------- #
# The deterministic fallback — same seven sections, no model involved
# --------------------------------------------------------------------------- #
def _join(items) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f'{", ".join(items[:-1])} and {items[-1]}'


def _count(value, noun: str = "mention") -> str:
    value = int(value or 0)
    return f"{value:,} {noun}{'' if value == 1 else 's'}"


def _theme(row) -> str:
    return f"{row['aspect']} ({row['net']:+.0f} net across {_count(row['mentions'])})"


def template_sections(pack: dict) -> tuple:
    """Write all seven sections from the pack deterministically.

    This is the floor the report can never fall below: it runs with no API key,
    no network and no model, and it obeys exactly the same honesty rails. The
    prose is plainer than Claude's and does not argue, but every number in it is
    computed from the same fact pack.
    """
    meta, sent = pack["meta"], pack["sentiment"]
    vol, trend = pack["volume"], pack["trend"]
    country = meta["country"] if meta["country"] != "All" else "the countries in view"
    n = vol["records"]
    out = {}

    if not n:
        blank = {"prose": ("No records match the current filters, so there is "
                           "nothing to analyse. Widen the year range, sentiment "
                           "or source filters and generate the report again."),
                 "data_notes": []}
        return {sid: dict(blank) for sid in SECTION_IDS}, "No records in view."

    reportable = [a for a in pack["aspects"] if a["reportable"]]
    loved = [a for a in reportable if a["net"] >= rails.NET_MARGIN][:3]
    disliked = [a for a in reportable if a["net"] <= -rails.NET_MARGIN][-3:][::-1]
    busiest = max(pack["aspects"], key=lambda a: a["mentions"], default=None)

    tone = ("strongly positive" if sent["net_sentiment"] >= 40 else
            "positive" if sent["net_sentiment"] >= 15 else
            "mixed but net-positive" if sent["net_sentiment"] > 0 else
            "mixed but net-negative" if sent["net_sentiment"] > -15 else "negative")
    headline = (f"{country} scores {sent['perception_score']}/100 on tourism "
                f"perception for {meta['period_label']} — {tone}."
                + (" Indicative only: the sample is below the reliability "
                   "threshold." if vol["is_thin"] else ""))

    # --- 01 Month in Review --------------------------------------------------
    paras = [
        f"Across {_count(n, 'record')} of {country} in {meta['period_label']}, "
        f"tourism perception scores <b>{sent['perception_score']} out of 100</b>, "
        f"on net sentiment of {sent['net_sentiment']:+.0f} points. The mix reads "
        f"as {tone}: {sent['positive_pct']:.0f}% of mentions are positive, "
        f"{sent['neutral_pct']:.0f}% neutral and {sent['negative_pct']:.0f}% "
        f"negative."]
    if loved or disliked:
        drivers = []
        if loved:
            drivers.append("the score is carried by " + _join(_theme(a) for a in loved))
        if disliked:
            drivers.append("and held back by " + _join(_theme(a) for a in disliked))
        paras.append(" ".join(drivers).capitalize() + ".")
    if trend.get("readable"):
        paras.append(
            f"Direction of travel is {trend['direction']}: perception moved from "
            f"{trend['first_perception']:.0f} in {trend['first_period']} to "
            f"{trend['last_perception']:.0f} in {trend['last_period']}, a change "
            f"of {trend['change_points']:+.0f} points. The priority for the "
            f"period is "
            + (f"defending the position on {loved[0]['aspect']} while closing the "
               f"gap on {disliked[0]['aspect']}." if loved and disliked else
               "consolidating the themes that already work and widening coverage "
               "on those that are too thinly evidenced to act on."))
    out["month_in_review"] = {"prose": "\n\n".join(paras), "data_notes": []}

    # --- 02 Perception Overview ----------------------------------------------
    paras = [
        f"On the 0–100 perception scale, where 50 is neutral, {country} scores "
        f"<b>{sent['perception_score']}</b>. That is built from "
        f"{sent['positive_records']:,} positive, {sent['neutral_records']:,} "
        f"neutral and {sent['negative_records']:,} negative records, giving net "
        f"sentiment of {sent['net_sentiment']:+.0f} points "
        f"({sent['positive_pct']:.0f}% positive against "
        f"{sent['negative_pct']:.0f}% negative)."]
    prov = pack["provenance"]
    if prov["sources"]:
        paras.append(
            "The reading is drawn from "
            + _join(facts.source_phrase(s) for s in prov["sources"])
            + (f". {prov['content_attributed_pct']:.0f}% of these were attributed "
               f"to {country} from what the text is about rather than the query "
               f"that surfaced them."
               if prov["content_attributed_pct"] is not None else "."))
    if prov["emotions"]:
        top = prov["emotions"][0]
        paras.append(
            f"Beyond neutral, <b>{top['emotion']}</b> is the dominant emotional "
            f"register, tagged on {_count(top['records'], 'record')} "
            f"({top['share_pct']:.0f}% of the view). Messaging that answers that "
            f"register will land harder than messaging that ignores it.")
    out["perception_overview"] = {"prose": "\n\n".join(paras), "data_notes": []}

    # --- 03 Thematic Analysis ------------------------------------------------
    paras = []
    if not pack["aspects"]:
        paras.append("No record in this view carries a theme tag, so there is no "
                     "thematic breakdown to report.")
    elif not reportable:
        paras.append(
            f"No theme reaches the {rails.MIN_ASPECT_MENTIONS}-mention floor "
            f"required to headline a finding. The most-discussed is "
            f"{busiest['aspect']}, raised in {_count(busiest['mentions'])}, "
            f"{busiest['positive_pct']:.0f}% of them positive — indicative only.")
    else:
        if loved:
            paras.append("Visitors are most positive about "
                         + _join(_theme(a) for a in loved)
                         + ". These are the themes with earned credibility: they "
                           "should lead the campaign, not compete with it.")
        if disliked:
            paras.append("Criticism concentrates on "
                         + _join(_theme(a) for a in disliked)
                         + ". Each is a fixable operational story before it is a "
                           "messaging problem — address the experience, then say "
                           "so publicly.")
        elif len(reportable) > 1:
            weakest = reportable[-2:][::-1]
            paras.append("No theme is decisively negative; the coolest reception "
                         "goes to " + _join(_theme(a) for a in weakest)
                         + ", which is where to watch next.")
        paras.append(
            f"{busiest['aspect'].capitalize()} is the most discussed theme "
            f"overall, raised in {_count(busiest['mentions'])} with "
            f"{busiest['positive_pct']:.0f}% of them positive.")
    voices = pack["voices"]
    if voices["negative"]:
        v = voices["negative"][0]
        paras.append(
            f"The most-engaged critical mention, from {v['author']} in "
            f"{v['date']}, carries {v['engagement']:,} likes — a single comment "
            f"with that reach shapes the impression of readers who never see the "
            f"aggregate.")
    out["thematic_analysis"] = {"prose": "\n\n".join(paras), "data_notes": []}

    # --- 04 Visitor Segments -------------------------------------------------
    seg = pack["segments"]
    notes = []
    if not seg.get("available"):
        paras = ["No visitor-segment signal is present in this view, so travel "
                 "styles cannot be compared."]
        notes.append(seg.get("note", ""))
    else:
        cohorts = seg["cohorts"]
        paras = [
            f"{seg['classified_pct']:.0f}% of records name a travel style — most "
            f"often " + _join(f"{c['segment']} ({_count(c['records'], 'record')})"
                              for c in cohorts[:3])
            + ". The remainder carry no style signal and are left unclassified "
              "rather than guessed at."]
        comparable = [c for c in cohorts if c["reportable"]]
        if len(comparable) >= 2:
            best = max(comparable, key=lambda c: c["perception_score"])
            worst = min(comparable, key=lambda c: c["perception_score"])
            paras.append(
                f"Among cohorts large enough to compare, <b>{best['segment']}</b> "
                f"travellers are the most positive at {best['perception_score']} "
                f"out of 100, and <b>{worst['segment']}</b> the most critical at "
                f"{worst['perception_score']}. Spend where the reception is "
                f"already warm and diagnose before spending where it is not.")
        else:
            paras.append(
                "No two cohorts clear the reliability threshold, so no cohort "
                "comparison is made here. Widening segment coverage is the "
                "prerequisite for segment-level targeting.")
    out["visitor_segments"] = {"prose": "\n\n".join(paras), "data_notes": notes}

    # --- 05 Competitive Benchmarking -----------------------------------------
    bench = pack["benchmark"]
    notes = []
    if not bench.get("available"):
        paras = ["Cross-country benchmarking is not available for this view."]
        notes.append(bench["note"])
    else:
        rank = bench.get("rank")
        paras = []
        if rank:
            paras.append(
                f"Against the {rank['of']} markets in the reference set that "
                f"clear the reliability threshold, {country} ranks "
                f"<b>{rank['position']} of {rank['of']}</b> on perception at "
                f"{sent['perception_score']} out of 100.")
        gaps = [g for g in bench["aspect_gaps"] if g["reportable"]]
        if gaps:
            ahead = [g for g in gaps if g["direction"] == "ahead"][:2]
            behind = [g for g in gaps if g["direction"] == "behind"][:2]
            if ahead:
                paras.append(
                    "Measured against the peer set, its clearest advantages are "
                    + _join(f"{g['aspect']} ({g['gap_points']:+.0f} points)"
                            for g in ahead)
                    + ". An advantage that peers do not share is the cheapest "
                      "thing to advertise.")
            if behind:
                paras.append(
                    "It trails the peer set on "
                    + _join(f"{g['aspect']} ({g['gap_points']:+.0f} points)"
                            for g in behind)
                    + ". These are the themes where a visitor comparing "
                      "destinations will pick someone else.")
        else:
            paras.append(
                f"No theme shows a gap wider than the {rails.GAP_MARGIN:.0f}-point "
                f"margin required before a cross-country difference is stated, so "
                f"the comparison stops at the overall score.")
            notes.append(
                f"Per-theme gaps against the peer set all fall inside the "
                f"{rails.GAP_MARGIN:.0f}-point margin and are therefore not "
                f"reported as differences.")
    out["competitive_benchmarking"] = {"prose": "\n\n".join(paras),
                                       "data_notes": notes}

    # --- 06 Trends & Signals -------------------------------------------------
    paras = []
    if trend.get("readable"):
        paras.append(
            f"Perception is {trend['direction']} across the period, moving from "
            f"{trend['first_perception']:.0f} in {trend['first_period']} to "
            f"{trend['last_perception']:.0f} in {trend['last_period']} "
            f"({trend['change_points']:+.0f} points). Conversation volume peaked "
            f"in {trend['peak_period']} at {_count(trend['peak_records'], 'record')}, "
            f"which is the window worth studying for what moved the needle.")
    else:
        paras.append(
            f"With {_count(n, 'record')} spread across "
            f"{_count(trend['period_count'], 'period')}, there is not enough "
            f"volume to separate a trend from drift. The trend chart is shown for "
            f"completeness only.")
    movement = [m for m in pack["aspect_movement"] if m["reportable"]]
    rising = [m for m in movement if m["direction"] == "worsening"][:2]
    easing = [m for m in movement if m["direction"] == "improving"][:2]
    if rising:
        paras.append(
            "Criticism is rising on "
            + _join(f"{m['aspect']} ({m['change_points']:+.0f} points of negative "
                    f"share between the first and second half of the period)"
                    for m in rising)
            + ". A worsening theme costs more to fix the longer it runs; this is "
              "the intervention to fund now.")
    if easing:
        paras.append(
            "Criticism eased on "
            + _join(f"{m['aspect']} ({m['change_points']:+.0f} points)"
                    for m in easing)
            + ". Whatever changed there is worth identifying and repeating.")
    if not rising and not easing:
        paras.append("No theme moved far enough between the halves of the period "
                     "to count as a signal rather than noise.")
    out["trends_signals"] = {"prose": "\n\n".join(paras), "data_notes": []}

    # --- 07 Recommendations --------------------------------------------------
    recs = []
    if disliked:
        worst = disliked[0]
        recs.append(
            f"<b>Fix and then message {worst['aspect']}.</b> It carries "
            f"{worst['net']:+.0f} net across {_count(worst['mentions'])}, the "
            f"weakest reception of any theme with enough evidence to act on. "
            f"Brief operators on the specific complaint, then put the remedy into "
            f"public communications — silence reads as denial.")
    elif reportable:
        # Nothing is outright criticised, but the weakest theme with enough
        # evidence behind it is still the one to watch.
        weakest = reportable[-1]
        recs.append(
            f"<b>Put {weakest['aspect']} on watch.</b> Nothing is decisively "
            f"criticised this period, but at {weakest['net']:+.0f} net across "
            f"{_count(weakest['mentions'])} it is the coolest reception of any "
            f"well-evidenced theme. Track it monthly; a theme is cheapest to fix "
            f"before it turns negative.")
    if loved:
        best = loved[0]
        recs.append(
            f"<b>Lead the next campaign with {best['aspect']}.</b> At "
            f"{best['net']:+.0f} net across {_count(best['mentions'])} it is the "
            f"most credible claim available, and credibility is what paid media "
            f"cannot buy.")
    if rising:
        recs.append(
            f"<b>Treat {rising[0]['aspect']} as the emerging risk.</b> Negative "
            f"share rose {rising[0]['change_points']:+.0f} points across the "
            f"period. Put it on the monthly watch list and set a review date "
            f"rather than waiting for the next report.")
    seg_cohorts = [c for c in pack["segments"].get("cohorts", []) if c["reportable"]]
    if len(seg_cohorts) >= 2:
        best_seg = max(seg_cohorts, key=lambda c: c["perception_score"])
        recs.append(
            f"<b>Weight acquisition spend toward {best_seg['segment']} "
            f"travellers.</b> They score {best_seg['perception_score']} out of "
            f"100 — the warmest cohort large enough to trust — so conversion "
            f"there costs least per visitor.")
    if bench.get("available"):
        gaps = [g for g in bench["aspect_gaps"] if g["reportable"]]
        behind = [g for g in gaps if g["direction"] == "behind"]
        if behind:
            recs.append(
                f"<b>Close the {behind[0]['aspect']} gap against the peer set.</b> "
                f"It sits {behind[0]['gap_points']:+.0f} points behind comparable "
                f"markets, which is where a switching visitor is lost.")
        rank = bench.get("rank") or {}
        if rank.get("position", 1) > 1 and rank.get("points_behind_leader"):
            recs.append(
                f"<b>Set the target against {rank['leader']}, not against last "
                f"period.</b> {country.capitalize()} ranks {rank['position']} of "
                f"{rank['of']} on perception, {rank['points_behind_leader']:.0f} "
                f"points behind the leader at {rank['leader_perception']} out of "
                f"100. That gap, not the month-on-month move, is what a visitor "
                f"choosing between destinations actually sees.")
    if vol["is_thin"]:
        recs.append(
            f"<b>Widen collection before acting.</b> This view holds "
            f"{n:,} records, under the {rails.MIN_SAMPLE} needed for a reliable "
            f"read. Extend the collection window before committing budget to any "
            f"finding above.")
    if not recs:
        recs.append(
            "No finding in this view clears the evidence thresholds needed to "
            "anchor a recommendation. The action for the period is to widen "
            "coverage until it does.")
    out["recommendations"] = {"prose": "\n\n".join(recs), "data_notes": []}

    # Every computed gap surfaces as a note somewhere, so nothing goes unsaid.
    placed = {note for s in out.values() for note in s["data_notes"]}
    orphans = [note for note in pack["data_notes"] if note not in placed]
    if orphans:
        out["month_in_review"]["data_notes"].extend(orphans)
    return out, headline


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #
class NarrativeResult:
    """The finished narrative plus an audit trail of how it was produced."""

    def __init__(self, sections: dict, headline: str, engine: str, log: list):
        self.sections = sections        # id -> {"prose", "data_notes"}
        self.headline = headline
        self.engine = engine            # "claude" | "claude+template" | "template"
        self.log = log                  # human-readable provenance lines

    def ordered(self):
        """[(number, title, prose, data_notes), ...] in report order."""
        out = []
        for sid, number, title, _ in SECTIONS:
            body = self.sections.get(sid)
            if body and body.get("prose"):
                out.append((number, title, body["prose"], body.get("data_notes", [])))
        return out


def _client():
    """An Anthropic client, or None when the SDK or the key is missing.

    The key is read from the environment by the SDK itself. It is never passed
    in, never read from a config file and never written to source.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "ANTHROPIC_API_KEY is not set"
    try:
        import anthropic
    except ImportError as exc:
        return None, f"anthropic SDK not installed ({exc})"
    try:
        return anthropic.Anthropic(), ""
    except Exception as exc:                                # pragma: no cover
        return None, f"could not create the Anthropic client ({exc})"


def _extract(response) -> dict | None:
    """Pull the submit_report payload out of a response.

    Prefers the tool call. Falls back to parsing a JSON object out of the text,
    because a model that answers in prose instead of calling the tool has still
    done the work and there is no reason to throw it away.
    """
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_report":
            return block.input
    text = "".join(b.text for b in response.content
                   if getattr(b, "type", None) == "text")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _as_sections(payload: dict) -> tuple:
    """Normalise the tool payload into {id: {prose, data_notes}} plus a headline."""
    sections = {}
    for item in (payload or {}).get("sections", []):
        sid = item.get("id")
        if sid not in SECTION_IDS:
            continue
        paragraphs = [p.strip() for p in str(item.get("prose", "")).split("\n\n")]
        sections[sid] = {
            "prose": "\n\n".join(p for p in paragraphs if p),
            "data_notes": [str(n).strip() for n in item.get("data_notes", [])
                           if str(n).strip()],
        }
    return sections, str((payload or {}).get("headline", "")).strip()


def _repair_prompt(problems: dict) -> str:
    lines = []
    for sid, values in problems.items():
        if not values:
            continue
        shown = ", ".join(f"{v:g}" for v in values)
        lines.append(f"  - {SECTION_TITLES[sid][1]} ({sid}): {shown}")
    return (
        "These figures appear in your draft but are not in the fact pack, so "
        "they cannot be published:\n\n" + "\n".join(lines) + "\n\n"
        "Rewrite the affected sections. For each figure above, either replace it "
        "with the pack figure it was meant to be, restate the point without a "
        "number, or drop the claim and add a DATA NOTE explaining that the "
        "figure is not available. Do not recalculate anything. Return all seven "
        "sections again by calling submit_report.")


def write(pack: dict, *, model: str = MODEL, max_repairs: int = 1) -> NarrativeResult:
    """Write the seven sections for a fact pack.

    Uses Claude when a key is available and falls back to `template_sections()`
    whenever it is not — or when the model's prose fails number verification and
    cannot be repaired. Never raises: a report that cannot be written well must
    still be written.
    """
    template, template_headline = template_sections(pack)
    log = []

    if not pack["volume"]["records"]:
        return NarrativeResult(template, template_headline, "template",
                               ["No records in view — nothing to narrate."])

    client, why = _client()
    if client is None:
        log.append(f"Narrative written from the built-in template: {why}.")
        return NarrativeResult(template, template_headline, "template", log)

    messages = [{"role": "user", "content": _user_prompt(pack)}]
    sections, headline, problems = {}, "", {}

    try:
        for attempt in range(max_repairs + 1):
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                tools=[REPORT_TOOL],
                messages=messages,
            )
            payload = _extract(response)
            if payload is None:
                log.append(f"Attempt {attempt + 1}: no usable response from {model}.")
                break
            sections, headline = _as_sections(payload)
            if not sections:
                log.append(f"Attempt {attempt + 1}: response carried no sections.")
                break
            problems = verify(sections, pack)
            failed = {k: v for k, v in problems.items() if v}
            if not failed:
                log.append(
                    f"Narrative written by {model}; all {len(sections)} sections "
                    f"passed number verification"
                    + (f" after {attempt} repair pass." if attempt else "."))
                break
            if attempt == max_repairs:
                log.append(
                    f"{len(failed)} section(s) still cited figures outside the "
                    f"computed set after repair and were replaced by the template.")
                break
            log.append(
                f"Attempt {attempt + 1}: unverified figures in "
                f"{', '.join(sorted(failed))} — sent back for repair.")
            # Continue the same conversation so the model keeps its draft in
            # context and only has to fix what failed. When it used the tool, the
            # correction has to come back as that tool's result; when it answered
            # in prose, a plain user turn is the only valid reply.
            tool_use_id = next((b.id for b in response.content
                                if getattr(b, "type", None) == "tool_use"), None)
            if tool_use_id:
                correction = {"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": _repair_prompt(failed),
                    "is_error": True,
                }]}
            else:
                correction = {"role": "user", "content": _repair_prompt(failed)}
            messages.append({"role": "assistant", "content": response.content})
            messages.append(correction)
    except Exception as exc:
        log.append(f"{model} call failed ({type(exc).__name__}: {exc}); "
                   f"fell back to the built-in template.")
        return NarrativeResult(template, template_headline, "template", log)

    if not sections:
        log.append("Fell back to the built-in template.")
        return NarrativeResult(template, template_headline, "template", log)

    # Splice: keep every verified section, substitute the template for the rest,
    # and fill any section the model simply did not return.
    final, substituted = {}, []
    for sid in SECTION_IDS:
        body = sections.get(sid)
        if body and body.get("prose") and not problems.get(sid):
            final[sid] = body
        else:
            final[sid] = template[sid]
            substituted.append(SECTION_TITLES[sid][0])
    engine = "claude" if not substituted else "claude+template"
    if substituted:
        log.append("Template used for section(s): " + ", ".join(substituted) + ".")
    return NarrativeResult(final, headline or template_headline, engine, log)
