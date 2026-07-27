"""SerpApi reviews provider settings.

Reads SERPAPI_KEY from the environment (never hard-coded) and holds the
per-country place search terms plus conservative caps for the free tier
(250 searches/month). Add a country by adding an entry to PLACES.

Keep each country's place list SMALL: every place costs up to ~2 SerpApi
searches per engine (one to resolve the place, one to pull its reviews), and
we pull from two engines (Google Hotels + Tripadvisor). With the defaults
below, one full run of the 5 South-African places uses at most
MAX_API_CALLS_PER_RUN searches — a hard ceiling enforced in adapter.py.
"""

import os


class SerpApiConfigError(RuntimeError):
    """Raised when the SerpApi key is missing so the caller can stop cleanly."""


def api_key() -> str:
    """Return SERPAPI_KEY from the environment, or raise a clear error.

    The key is read ONLY from the environment and is never hard-coded or logged.
    """
    key = os.environ.get("SERPAPI_KEY")
    if not key:
        raise SerpApiConfigError(
            "Missing SERPAPI_KEY environment variable. Set it in your shell or a "
            "local .env file (never commit the key). Get one at serpapi.com."
        )
    return key


# Per-country place search terms — a handful of well-known hotels/attractions.
# SMALL by design to respect the 250-searches/month free tier.
PLACES = {
    "South Africa": [
        "Table Mountain",
        "V&A Waterfront hotels Cape Town",
        "Kruger National Park lodges",
        "hotels Johannesburg",
        "hotels Durban",
    ],
    "Nigeria": [
        "Eko Hotel and Suites Lagos",
        "Transcorp Hilton Abuja",
        "Lekki Conservation Centre Lagos",
        "hotels Victoria Island Lagos",
        "Obudu Mountain Resort Nigeria",
    ],
    "Ghana": [
        "Cape Coast Castle Ghana",
        "Kempinski Hotel Gold Coast City Accra",
        "Kakum National Park Ghana",
        "Labadi Beach Hotel Accra",
        "hotels Kumasi Ghana",
    ],
    "Zimbabwe": [
        "Victoria Falls Zimbabwe",
        "Victoria Falls Hotel Zimbabwe",
        "Hwange National Park lodges",
        "Great Zimbabwe National Monument",
        "hotels Harare Zimbabwe",
    ],
}

# Which SerpApi review engines to pull from for each place.
ENGINES = ("google_hotels", "tripadvisor")

# --- Hard safety caps (protect the 250/month free quota) -------------------
# The adapter counts EVERY SerpApi search against this ceiling and stops the run
# the moment it is reached, so an accidental large place list can never blow the
# monthly free limit.
MAX_API_CALLS_PER_RUN = 20     # never issue more than this many searches per run
REVIEWS_PER_PLACE = 20         # cap reviews mapped per place per engine (1 page)

# SerpApi locale params.
GL = "us"                      # geo
HL = "en"                      # language
CURRENCY = "USD"


def places_for(country: str):
    if country not in PLACES:
        raise KeyError(
            f"No SerpApi place terms configured for {country!r}. "
            f"Add an entry to providers/serpapi_reviews/config.py PLACES."
        )
    return PLACES[country]
