"""Reddit-specific settings: subreddits to search per country.

Add a new country by adding an entry to SUBREDDITS; countries without an entry
fall back to DEFAULT_SUBREDDITS.
"""

# A travel-focused default set, plus "all" for the widest net.
DEFAULT_SUBREDDITS = ["travel", "solotravel", "TravelNoFilter", "all"]

# Per-country overrides (optional).
SUBREDDITS = {
    "South Africa": ["travel", "solotravel", "southafrica", "all"],
    "Rwanda": ["travel", "solotravel", "Rwanda", "africa", "all"],
}

MAX_RESULTS_DEFAULT = 100  # max posts per subreddit


def subreddits_for(country: str):
    return SUBREDDITS.get(country, DEFAULT_SUBREDDITS)
