"""YouTube-specific settings: per-country search queries, year span, quota caps.

Add a new country by adding an entry to QUERIES (and optionally REGION_CODES).
No code changes needed elsewhere.
"""

# Tourism-specific search queries per country. These target travel/destination
# content (guides, vlogs, itineraries, safety) rather than generic mentions, so
# the comments we pull are more likely to be actual tourism sentiment.
QUERIES = {
    "South Africa": [
        "Cape Town travel guide",
        "things to do in South Africa",
        "is South Africa safe for tourists",
        "South Africa travel vlog",
        "visiting Johannesburg",
        "Kruger safari",
        "South Africa itinerary",
        "Garden Route road trip",
        "Cape Town vlog",
        "South Africa holiday",
        "best places to visit South Africa",
        "Durban travel",
        "South Africa tourist tips",
    ],
    "Rwanda": [
        "Rwanda travel guide",
        "things to do in Rwanda",
        "is Rwanda safe for tourists",
        "Rwanda travel vlog",
        "visiting Kigali",
        "gorilla trekking Rwanda",
        "Rwanda itinerary",
        "Volcanoes National Park Rwanda",
        "Kigali vlog",
        "Rwanda holiday",
        "best places to visit Rwanda",
        "Lake Kivu travel",
        "Rwanda tourist tips",
    ],
}

# Optional ISO 3166-1 alpha-2 region bias per country (improves relevance).
REGION_CODES = {
    "South Africa": "ZA",
    "Rwanda": "RW",
}

# --- Year sweep ------------------------------------------------------------
# Comments can't be filtered by year, but a video's comments cluster around when
# it was posted. So we sweep travel videos published in EACH year and collect
# their comments, spreading comment dates across the whole timeline.
YEAR_START = 2015
YEAR_END = 2026

# --- Quota / volume caps ---------------------------------------------------
# search.list costs 100 units/call; commentThreads.list costs 1 unit/call;
# the default daily quota is ~10,000 units. Bound the number of SEARCH calls:
#   search calls = SEARCH_QUERIES_PER_YEAR * (YEAR_END - YEAR_START + 1)
# Default: 3 * 12 = 36 calls = 3,600 units, leaving plenty of headroom.
SEARCH_QUERIES_PER_YEAR = 3     # how many queries to search per year (rotated)
VIDEOS_PER_QUERY_PER_YEAR = 5   # videos taken per (query, year)

# Comments per video, pulled in BOTH orders (relevance + chronological "time"),
# so a single popular video can't dominate and we capture a wider date spread.
COMMENTS_PER_VIDEO_PER_ORDER = 40

# --- Transcripts -----------------------------------------------------------
# Transcripts (via youtube-transcript-api, no API key) are an extra signal
# alongside comments. Each video's transcript is split into ~chunk-sized pieces.
INCLUDE_TRANSCRIPTS = True
TRANSCRIPT_CHUNK_CHARS = 280        # target size of each transcript chunk
TRANSCRIPT_LANGUAGES = ("en", "en-US", "en-GB")
MAX_TRANSCRIPT_CHUNKS_PER_VIDEO = 40  # cap so one long video can't dominate

# Overall safety cap on comments per run. None = no cap (rely on the per-year /
# per-video limits), which keeps the date spread even across all years.
MAX_RESULTS_DEFAULT = None


def queries_for(country: str):
    if country not in QUERIES:
        raise KeyError(
            f"No YouTube queries configured for {country!r}. "
            f"Add an entry to providers/youtube/config.py QUERIES."
        )
    return QUERIES[country]


def years():
    return list(range(YEAR_START, YEAR_END + 1))
