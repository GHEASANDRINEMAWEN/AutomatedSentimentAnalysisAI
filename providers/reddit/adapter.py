"""Reddit provider adapter (PRAW). Pull + map only — no scoring, no filtering.

Moved from the original top-level collect_reddit.py. Searches a set of
subreddits for tourism chatter about a country and maps each submission into
the shared common record. Credentials come from environment variables via the
top-level config module and are never hard-coded.

Field mapping (submission -> common record):
    title + selftext        -> text
    author                  -> author
    score                   -> engagement
    created_utc             -> timestamp  (the post's publish date)
    permalink               -> url        (direct link to the post)
    id                      -> source_id
    "reddit"                -> source
"""

import config  # top-level: Reddit API credentials from environment variables

import praw

from core.record import Record, to_iso8601
from providers.reddit import config as reddit_config


def _build_query(country: str) -> str:
    """A simple tourism-intent query for a country."""
    return f"{country} (tourism OR travel OR visit OR vacation OR holiday OR safari)"


def _make_client() -> "praw.Reddit":
    config.validate()
    return praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
        check_for_async=False,
    )


def fetch(country: str, queries=None, max_results=None):
    """Pull Reddit submissions for `country` and map them to common records.

    Args:
        country: country name.
        queries: optional override of the subreddits to search.
        max_results: max posts per subreddit (defaults to config value).

    Returns a list of common-record dicts (sentiment not yet attached).
    """
    subreddits = queries or reddit_config.subreddits_for(country)
    if max_results is None:
        max_results = reddit_config.MAX_RESULTS_DEFAULT

    reddit = _make_client()
    query = _build_query(country)

    seen = set()
    records = []
    for sub in subreddits:
        try:
            results = reddit.subreddit(sub).search(query, sort="relevance", limit=max_results)
            for post in results:
                if post.id in seen:
                    continue
                seen.add(post.id)
                title = post.title or ""
                body = post.selftext or ""
                text = (title + "\n\n" + body).strip() if body else title
                record = Record(
                    source="reddit",
                    source_id=post.id,
                    country=country,
                    text=text,
                    author=str(post.author) if post.author else "",
                    timestamp=to_iso8601(post.created_utc),
                    url=f"https://reddit.com{post.permalink}",
                    engagement=int(post.score or 0),
                )
                records.append(record.to_dict())
        except Exception as exc:  # keep going if one subreddit fails
            print(f"  ! search failed in r/{sub}: {exc}")

    return records
