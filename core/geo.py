"""Country attribution — decide which country a record is actually ABOUT.

A record arrives tagged with the country whose search query pulled it. That is
only a hint: a "Kenya travel" search surfaces videos about Zanzibar, and a
comment under a Nigeria vlog may be entirely about Ghana. Trusting the query
mislabels those rows across countries.

This module re-derives `country` from evidence, in three tiers:

  1. content  — the record's own text names places/demonyms of a country.
  2. context  — the text is place-less ("stunning!"), but the other records
                from the SAME video collectively point at one country.
  3. query    — no evidence anywhere; keep the country that pulled the record.

`country_query` preserves the original pull country and `country_source` records
which tier decided, so attribution stays auditable and the pass is IDEMPOTENT:
assign() always re-derives from `country_query`, never from a previous verdict.

Strength is measured in DISTINCT matched terms, not occurrences, so a comment
repeating "Egypt Egypt Egypt" does not outweigh one naming Nairobi, Mombasa and
the Maasai Mara. Ties are resolved in favour of the pull country: we only move a
record when another country is strictly better evidenced.

SWAPPABLE BY DESIGN: the public contract is `assign(records) -> records`. An
LLM/NER-based attributor can implement the same signature and drop in.
"""

import re
from collections import Counter

# Terms that identify a country: its name, demonym, capital, major cities and
# well-known landmarks. Deliberately CONSERVATIVE — a term that also names a
# place (or a thing) elsewhere is omitted rather than risk a wrong reassignment:
#   "benin"    -> the neighbouring country, not Benin City  -> only "benin city"
#   "tamale"   -> the Mexican dish as often as the Ghanaian city -> qualified
#   "fez"      -> the hat -> qualified
#   "sahara"   -> spans Morocco, Egypt, Algeria, Tunisia -> omitted
#   "nyanga"   -> Zimbabwe's highlands AND a Cape Town township -> omitted
#   "yoruba"   -> identity/diaspora chatter far more than travel -> omitted
# ASCII terms are matched word-bounded (so "kano" cannot fire inside "volcano"
# and "nile" cannot fire inside "juvenile"); non-ASCII terms (Arabic) are matched
# as substrings because Arabic attaches prefixes and would defeat \b.
GAZETTEER = {
    "South Africa": (
        "south africa", "south african", "cape town", "capetown",
        "johannesburg", "joburg", "jozi", "durban", "pretoria", "soweto",
        "kruger", "table mountain", "garden route", "stellenbosch",
        "drakensberg", "knysna", "port elizabeth", "gqeberha", "western cape",
        "robben island", "sun city", "hermanus", "plettenberg", "cape point",
        "boulders beach", "bloemfontein", "mpumalanga", "kwazulu", "afrikaans",
    ),
    "Rwanda": (
        "rwanda", "rwandan", "kigali", "volcanoes national park", "nyungwe",
        "lake kivu", "akagera", "musanze", "ruhengeri", "gisenyi", "rubavu",
        "bisoke", "karisimbi", "huye", "kinyarwanda",
        "land of a thousand hills",
    ),
    "Kenya": (
        "kenya", "kenyan", "nairobi", "mombasa", "maasai mara", "masai mara",
        "diani", "lamu", "amboseli", "lake nakuru", "nakuru", "tsavo",
        "malindi", "mount kenya", "watamu", "samburu", "kisumu", "naivasha",
    ),
    "Tanzania": (
        "tanzania", "tanzanian", "zanzibar", "serengeti", "kilimanjaro",
        "ngorongoro", "dar es salaam", "arusha", "stone town", "dodoma",
        "mafia island", "tarangire", "pemba island", "selous", "nungwi",
        "mikumi",
    ),
    "Egypt": (
        "egypt", "egyptian", "cairo", "giza", "pyramids", "sphinx", "luxor",
        "aswan", "nile", "felucca", "sharm el sheikh", "sharm el-sheikh",
        "hurghada", "alexandria", "red sea", "valley of the kings",
        "abu simbel", "dahab", "siwa", "karnak", "khan el khalili",
        "marsa alam",
        "مصر", "القاهرة", "الأهرامات", "الاهرامات", "الجيزة", "الأقصر", "أسوان",
        "شرم الشيخ", "الغردقة", "الإسكندرية", "البحر الأحمر",
    ),
    "Senegal": (
        "senegal", "sénégal", "senegalese", "sénégalais", "dakar", "goree",
        "gorée", "lac rose", "lake retba", "casamance", "sine saloum",
        "ziguinchor", "thiès", "djoudj", "african renaissance monument",
        "wolof",
    ),
    "Ghana": (
        "ghana", "ghanaian", "accra", "kumasi", "cape coast", "kakum",
        "elmina", "takoradi", "labadi", "mole national park", "lake volta",
        "volta region", "osu castle", "aburi", "kotoka", "ashanti", "akwaaba",
        "tamale ghana",
    ),
    "Morocco": (
        "morocco", "moroccan", "maroc", "marrakech", "marrakesh", "casablanca",
        "chefchaouen", "rabat", "tangier", "essaouira", "merzouga",
        "atlas mountains", "agadir", "meknes", "ouarzazate", "jemaa el",
        "hassan ii", "ait ben haddou", "riad", "tagine", "fez morocco",
        "fes morocco",
        "المغرب", "مراكش", "الدار البيضاء", "الرباط", "طنجة",
    ),
    "Cameroon": (
        "cameroon", "cameroun", "cameroonian", "camerounais", "yaounde",
        "yaoundé", "douala", "kribi", "mount cameroon", "limbe", "bamenda",
        "buea", "foumban", "waza", "korup", "dschang",
    ),
    "Nigeria": (
        "nigeria", "nigerian", "naija", "lagos", "abuja", "calabar",
        "benin city", "ibadan", "port harcourt", "kano", "yankari", "obudu",
        "olumo rock", "zuma rock", "idanre", "lekki", "badagry", "niger delta",
        "oshogbo", "osogbo", "eko atlantic", "erin ijesha", "nollywood",
        "abeokuta", "enugu", "kaduna", "aso rock",
    ),
    "Zimbabwe": (
        "zimbabwe", "zimbabwean", "harare", "victoria falls", "vic falls",
        "bulawayo", "hwange", "matobo", "matopos", "mana pools", "lake kariba",
        "kariba", "eastern highlands", "chimanimani", "gonarezhou", "mutare",
        "masvingo", "chinhoyi", "great zimbabwe", "shona", "zambezi",
    ),
}


def _compile(terms):
    """Return (word-bounded regex or None, tuple of substring-only terms)."""
    ascii_terms = [t for t in terms if t.isascii()]
    other_terms = tuple(t for t in terms if not t.isascii())
    pattern = None
    if ascii_terms:
        # Longest-first so "cape coast" is preferred over a shorter overlap.
        alts = "|".join(re.escape(t) for t in sorted(ascii_terms, key=len, reverse=True))
        pattern = re.compile(r"\b(?:" + alts + r")\b")
    return pattern, other_terms


_MATCHERS = {country: _compile(terms) for country, terms in GAZETTEER.items()}

# "Greetings from Morocco" under a Tanzania safari video says where the VIEWER
# is, not what the video is about. Mentions introduced by a "from" marker are
# therefore ignored — including the Arabic ‏من‎ , since whole comment threads on
# Arabic-language travel videos are viewer-origin greetings and would otherwise
# drag the video's entire comment section to the wrong country.
_ORIGIN_MARKER = re.compile(
    r"(?:\bfrom\s+(?:the\s+|a\s+|an\s+|my\s+|your\s+)?|من\s+(?:ال)?)$"
)
# How far back to look for that marker (enough for "from the ", not a sentence).
_ORIGIN_LOOKBEHIND = 16

# Before a video's pooled evidence may re-home its place-less comments, the
# leading country must clear all three bars: enough distinct terms of its own,
# contributed by more than one record, and a dominant share of the video's total
# evidence. A video's whole comment section rides on this verdict, so thin or
# contested evidence must not be enough to move it.
MIN_CONTEXT_EVIDENCE = 5
MIN_CONTEXT_RECORDS = 2
MIN_CONTEXT_SHARE = 0.7


def _is_origin_mention(lowered: str, start: int) -> bool:
    """True when the match at `start` is preceded by a viewer-origin marker."""
    return bool(_ORIGIN_MARKER.search(lowered[max(0, start - _ORIGIN_LOOKBEHIND):start]))


def evidence(text: str) -> Counter:
    """Distinct gazetteer terms per country found in `text`.

    Counting DISTINCT terms (not occurrences) keeps a repeated country name from
    outweighing a comment that names several places of another country. Mentions
    that only state where the commenter is writing from are not evidence.
    """
    scores = Counter()
    if not text:
        return scores
    lowered = text.lower()
    for country, (pattern, substrings) in _MATCHERS.items():
        hits = set()
        if pattern:
            for match in pattern.finditer(lowered):
                if not _is_origin_mention(lowered, match.start()):
                    hits.add(match.group(0))
        for term in substrings:
            start = lowered.find(term)
            while start != -1:
                if not _is_origin_mention(lowered, start):
                    hits.add(term)
                    break
                start = lowered.find(term, start + 1)
        if hits:
            scores[country] = len(hits)
    return scores


def _winner(scores: Counter, fallback: str):
    """Best-evidenced country, or None if empty / not strictly better.

    `fallback` (the pull country) wins ties, so a record only moves when another
    country is unambiguously better evidenced.
    """
    if not scores:
        return None
    best = max(scores.values())
    leaders = sorted(c for c, v in scores.items() if v == best)
    if fallback in leaders:
        return fallback
    return leaders[0] if len(leaders) == 1 else None


def video_key(rec: dict):
    """Group key for records sharing a video (YouTube comments + transcript).

    Returns None for sources that have no video grouping (reviews, Reddit), so
    they never pool evidence with unrelated rows.
    """
    if rec.get("source") not in ("youtube", "youtube_transcript"):
        return None
    url = rec.get("url") or ""
    if "v=" not in url:
        return None
    return url.split("v=", 1)[1].split("&", 1)[0]


def assign(records):
    """Set `country`, `country_query` and `country_source` on each record.

    Idempotent: attribution is always re-derived from `country_query` (seeded
    once from the incoming `country`), so re-running never compounds a previous
    verdict. Returns the same list.
    """
    # Seed the immutable pull country, then score every record's own text once.
    own = []
    for rec in records:
        if not rec.get("country_query"):
            rec["country_query"] = rec.get("country") or ""
        own.append(evidence(rec.get("text") or ""))

    # Pool each video's evidence so place-less comments inherit their video's
    # subject rather than the search query that happened to surface it.
    pooled = {}
    contributors = Counter()
    for rec, scores in zip(records, own):
        key = video_key(rec)
        if key is None or not scores:
            continue
        pooled.setdefault(key, Counter()).update(scores)
        contributors[key] += 1

    for rec, scores in zip(records, own):
        pulled = rec.get("country_query") or ""
        if scores:
            winner = _winner(scores, pulled)
            if winner:
                rec["country"], rec["country_source"] = winner, "content"
                continue
        key = video_key(rec)
        context = pooled.get(key)
        if context and contributors[key] >= MIN_CONTEXT_RECORDS:
            winner = _winner(context, pulled)
            if (winner
                    and context[winner] >= MIN_CONTEXT_EVIDENCE
                    and context[winner] / sum(context.values()) >= MIN_CONTEXT_SHARE):
                rec["country"], rec["country_source"] = winner, "context"
                continue
        rec["country"], rec["country_source"] = pulled, "query"
    return records


def summary(records) -> dict:
    """Attribution stats: per-tier counts and how many records moved country."""
    tiers = Counter(r.get("country_source") or "query" for r in records)
    moved = Counter()
    for rec in records:
        pulled = rec.get("country_query")
        if pulled and rec.get("country") != pulled:
            moved[f'{pulled} -> {rec["country"]}'] += 1
    return {"tiers": dict(tiers), "moved": moved, "moved_total": sum(moved.values())}
