"""YouTube-specific settings: per-country search queries and quota caps.

Add a new country by adding an entry to QUERIES (and optionally REGION_CODES).
No code changes needed elsewhere.
"""

# Search queries per country. Start with South Africa.
QUERIES = {
    "South Africa": [
        "South Africa travel",
        "visit South Africa",
        "South Africa tourism",
    ],
}

# Optional ISO 3166-1 alpha-2 region bias per country (improves relevance).
REGION_CODES = {
    "South Africa": "ZA",
}

# Quota caps — keep these modest so a run doesn't burn the daily quota.
# search.list costs 100 units/call; commentThreads.list costs 1 unit/call;
# the default daily quota is ~10,000 units.
MAX_VIDEOS_PER_QUERY = 12      # videos returned per search query
MAX_COMMENTS_PER_VIDEO = 200   # top-level comments per video (paged, 100/page)
MAX_RESULTS_DEFAULT = 1500     # overall cap on comments pulled per run


def queries_for(country: str):
    if country not in QUERIES:
        raise KeyError(
            f"No YouTube queries configured for {country!r}. "
            f"Add an entry to providers/youtube/config.py QUERIES."
        )
    return QUERIES[country]
