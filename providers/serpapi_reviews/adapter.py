"""SerpApi reviews provider adapter — Google Hotels + Tripadvisor reviews.

Pull + map only (no scoring/filtering; relevance, sentiment, aspects and emotion
all happen in core/, the same as every other provider). Two signals, both mapped
into the shared common record:

  * Google Hotels reviews  (source="google_hotels")
        engine=google_hotels        -> resolve a place to a property_token
        engine=google_hotels_reviews -> pull that property's reviews
  * Tripadvisor reviews    (source="tripadvisor")
        engine=tripadvisor          -> resolve a place to a location token
        engine=tripadvisor_reviews  -> pull that location's reviews

Free-tier safety: EVERY SerpApi search is counted against a hard per-run ceiling
(config.MAX_API_CALLS_PER_RUN) and logged, and the run stops the moment the cap
is reached — so we never blow the 250-searches/month free limit by accident. The
number of searches used is exposed via `LAST_RUN` for reporting.

SERPAPI_KEY is read only from the environment (via config.api_key()); it is never
hard-coded. Dedupe is on source_id (the review id, else its link).

Field mapping (review -> common record):
    review text/snippet   -> text
    reviewer name         -> author
    review date           -> timestamp
    helpful/votes else 0  -> engagement
    review link           -> url
    review id, else link  -> source_id
    1-5 star rating       -> rating
    engine name           -> source ("google_hotels" | "tripadvisor")
"""

import datetime as _dt

import requests

from core.record import Record, to_iso8601
from providers.serpapi_reviews import config as sa_config

SERPAPI_ENDPOINT = "https://serpapi.com/search"

# Populated by fetch() so callers (run.py) can report searches used this run.
LAST_RUN = {"searches_used": 0, "limit": sa_config.MAX_API_CALLS_PER_RUN}


class SerpApiError(RuntimeError):
    """A SerpApi request failed; the caller skips that place and continues."""


class BudgetExhausted(SerpApiError):
    """The hard per-run search cap has been reached."""


class SearchBudget:
    """Counts SerpApi searches and enforces a hard per-run ceiling."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def remaining(self) -> int:
        return self.limit - self.used

    def spend(self, engine: str) -> None:
        # Count BEFORE issuing the call: a call that errors still counts, which
        # keeps us safely under the monthly quota even on failures.
        if self.used >= self.limit:
            raise BudgetExhausted(f"per-run cap of {self.limit} searches reached")
        self.used += 1
        print(f"  [serpapi] search #{self.used}/{self.limit} (engine={engine})")


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
def _serpapi_search(engine: str, params: dict, budget: SearchBudget) -> dict:
    """Run one SerpApi search for `engine`, counting it against `budget`.

    Prefers the official `serpapi` client when installed, otherwise calls the
    documented HTTP endpoint with `requests` (which the client also wraps), so
    the provider has no hard dependency on the client library.
    """
    budget.spend(engine)
    full = dict(params, engine=engine, api_key=sa_config.api_key(), output="json")

    try:
        import serpapi as _client  # optional; requirements-pipeline.txt
    except ImportError:
        _client = None
    if _client is not None and hasattr(_client, "search"):
        try:
            return dict(_client.search(**full))
        except Exception as exc:  # normalize to our error type
            raise SerpApiError(str(exc)) from exc

    resp = requests.get(SERPAPI_ENDPOINT, params=full, timeout=60)
    if resp.status_code != 200:
        raise SerpApiError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    if data.get("error"):
        raise SerpApiError(str(data["error"]))
    return data


# --------------------------------------------------------------------------- #
# Review-field extraction (defensive: SerpApi shapes vary by engine/version)
# --------------------------------------------------------------------------- #
def _first(d: dict, keys, default=""):
    for k in keys:
        v = d.get(k)
        if v not in (None, "", []):
            return v
    return default


def _extract_rating(raw: dict):
    """Return a float 1-5 star rating, or None."""
    r = raw.get("rating")
    if isinstance(r, dict):
        r = r.get("rating") or r.get("value")
    if r is None:
        r = raw.get("overall_rating") or raw.get("stars") or raw.get("score")
    try:
        return round(float(r), 1) if r is not None else None
    except (TypeError, ValueError):
        return None


def _extract_engagement(raw: dict) -> int:
    """Return helpful/votes count if present, else 0."""
    for k in ("helpful_votes", "helpful", "likes", "votes", "thumbs_up",
              "upvotes", "helpful_count"):
        v = raw.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
    return 0


_DATE_FORMATS = (
    "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%B %d, %Y", "%b %d, %Y",
    "%d %B %Y", "%d %b %Y", "%B %Y", "%b %Y", "%m/%d/%Y",
)


def _normalize_date(value) -> str:
    """Best-effort normalize a review date to ISO 8601 UTC.

    Review dates arrive in many shapes ("2023-08-01", "August 2023", "Aug 2023").
    Anything we can't parse is passed through for to_iso8601 to handle.
    """
    if not value:
        return ""
    s = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return (_dt.datetime.strptime(s, fmt)
                    .replace(tzinfo=_dt.timezone.utc).isoformat())
        except ValueError:
            continue
    return s


def _map_review(raw: dict, source: str, country: str):
    """Map one raw SerpApi review dict into a common-record dict, or None."""
    text = _first(raw, ("snippet", "text", "review", "review_text",
                        "description", "content"))
    text = " ".join(str(text).split())
    if not text:
        return None

    author = raw.get("user")
    if isinstance(author, dict):
        author = author.get("name") or author.get("username") or ""
    if not author:
        author = _first(raw, ("author", "name", "reviewer", "username"))

    link = _first(raw, ("link", "review_link", "url", "source_url"))
    link = link if isinstance(link, str) else ""
    review_id = _first(raw, ("review_id", "id", "review_token")) or link
    if not review_id:
        # Last-resort stable key so re-runs still dedupe on identical reviews.
        review_id = f"{source}:{author}:{text[:60]}"

    record = Record(
        source=source,
        source_id=str(review_id),
        country=country,
        text=text,
        author=str(author or ""),
        timestamp=to_iso8601(_normalize_date(
            _first(raw, ("date", "published_date", "review_date",
                         "created_at", "time")))),
        url=link,
        engagement=_extract_engagement(raw),
        rating=_extract_rating(raw),
    )
    return record.to_dict()


# --------------------------------------------------------------------------- #
# Per-engine pulls
# --------------------------------------------------------------------------- #
def _hotel_stay_dates():
    """(check_in, check_out) ~30 days out — required by the google_hotels engine."""
    start = _dt.date.today() + _dt.timedelta(days=30)
    return start.isoformat(), (start + _dt.timedelta(days=1)).isoformat()


def _google_hotels_reviews(place: str, country: str, budget: SearchBudget):
    """Resolve `place` to a property_token, then pull its reviews."""
    check_in, check_out = _hotel_stay_dates()
    try:
        search = _serpapi_search("google_hotels", {
            "q": place, "check_in_date": check_in, "check_out_date": check_out,
            "adults": 2, "currency": sa_config.CURRENCY,
            "gl": sa_config.GL, "hl": sa_config.HL,
        }, budget)
    except SerpApiError as exc:
        print(f"  ! google_hotels search failed for {place!r}: {exc}")
        return []

    token, name = None, place
    for prop in (search.get("properties") or []):
        if prop.get("property_token"):
            token = prop["property_token"]
            name = prop.get("name", place)
            break
    if not token:
        print(f"  ! no property_token for {place!r} (google_hotels)")
        return []

    try:
        reviews = _serpapi_search("google_hotels_reviews", {
            "property_token": token, "gl": sa_config.GL, "hl": sa_config.HL,
        }, budget)
    except SerpApiError as exc:
        print(f"  ! google_hotels_reviews failed for {name!r}: {exc}")
        return []

    out = []
    for raw in (reviews.get("reviews") or [])[: sa_config.REVIEWS_PER_PLACE]:
        rec = _map_review(raw, source="google_hotels", country=country)
        if rec:
            out.append(rec)
    return out


def _tripadvisor_reviews(place: str, country: str, budget: SearchBudget):
    """Resolve `place` to a Tripadvisor location, then pull its reviews."""
    try:
        search = _serpapi_search("tripadvisor", {
            "q": place, "gl": sa_config.GL, "hl": sa_config.HL,
        }, budget)
    except SerpApiError as exc:
        print(f"  ! tripadvisor search failed for {place!r}: {exc}")
        return []

    # SerpApi shapes vary; look for a location token under the common keys.
    candidates = (search.get("locations") or search.get("location_results")
                  or search.get("results") or search.get("organic_results") or [])
    token, name = None, place
    for cand in candidates:
        tok = _first(cand, ("location_id", "data_id", "property_token", "token"))
        if tok:
            token = tok
            name = _first(cand, ("name", "title"), place)
            break
    if not token:
        print(f"  ! no Tripadvisor location token for {place!r}")
        return []

    try:
        reviews = _serpapi_search("tripadvisor_reviews", {
            "location_id": token, "gl": sa_config.GL, "hl": sa_config.HL,
        }, budget)
    except SerpApiError as exc:
        print(f"  ! tripadvisor_reviews failed for {name!r}: {exc}")
        return []

    out = []
    for raw in (reviews.get("reviews") or [])[: sa_config.REVIEWS_PER_PLACE]:
        rec = _map_review(raw, source="tripadvisor", country=country)
        if rec:
            out.append(rec)
    return out


_ENGINE_FUNCS = {
    "google_hotels": _google_hotels_reviews,
    "tripadvisor": _tripadvisor_reviews,
}


# --------------------------------------------------------------------------- #
# Public contract
# --------------------------------------------------------------------------- #
def fetch(country, queries=None, max_results=None):
    """Pull Google Hotels + Tripadvisor reviews for `country` as common records.

    Args:
        country: country name; must have place terms configured unless `queries`
                 is passed explicitly.
        queries: optional override of the per-country place search terms.
        max_results: optional overall cap on reviews returned (defaults to no
                     cap beyond the per-place / per-engine limits).

    Returns a list of common-record dicts (no sentiment / relevance yet),
    mixing source="google_hotels" and source="tripadvisor". Every SerpApi
    search is counted against a hard per-run ceiling.
    """
    places = queries or sa_config.places_for(country)
    sa_config.api_key()  # fail fast with a clear message if the key is missing
    budget = SearchBudget(sa_config.MAX_API_CALLS_PER_RUN)
    print(f"  SerpApi reviews for {country!r}: {len(places)} places x "
          f"{len(sa_config.ENGINES)} engines, hard cap {budget.limit} searches.")

    records, seen = [], set()
    stop = False
    for place in places:
        if stop or budget.remaining() <= 0:
            break
        for engine in sa_config.ENGINES:
            # Each engine needs ~2 searches (resolve + reviews); don't start one
            # we can't finish within the cap.
            if budget.remaining() < 2:
                print(f"  ! <2 searches left — skipping {engine} for {place!r}.")
                continue
            try:
                got = _ENGINE_FUNCS[engine](place, country, budget)
            except BudgetExhausted:
                print("  ! per-run search cap reached — stopping.")
                stop = True
                break
            new = 0
            for rec in got:
                if rec["source_id"] in seen:
                    continue
                seen.add(rec["source_id"])
                records.append(rec)
                new += 1
                if max_results is not None and len(records) >= max_results:
                    break
            print(f"  {engine:<16} {place!r}: +{new} reviews "
                  f"({len(records)} total)")
            if max_results is not None and len(records) >= max_results:
                stop = True
                break

    LAST_RUN["searches_used"] = budget.used
    LAST_RUN["limit"] = budget.limit
    print(f"  SerpApi searches used this run: {budget.used}/{budget.limit} "
          f"(free tier budget: 250/month)")
    return records
