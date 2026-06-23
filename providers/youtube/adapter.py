"""YouTube Data API v3 provider adapter.

Searches travel/tourism videos per country and pulls top-level comments via the
commentThreads endpoint, mapping each comment into the shared common record.
Pull + map only — no scoring, no filtering (relevance is marked later in core).

To spread comment dates across the whole timeline, videos are searched per year
using the API's publishedAfter/publishedBefore filters; a video's comments
cluster around when it was posted. Each video's comments are pulled in both
"relevance" and "time" order so one popular video can't dominate.

The API key is read only from the YOUTUBE_API_KEY environment variable; it is
never hard-coded.

Field mapping (comment -> common record):
    textDisplay        -> text
    authorDisplayName  -> author
    likeCount          -> engagement
    publishedAt        -> timestamp   (the comment's publish date)
    watch?v=..&lc=..   -> url         (direct link to the comment)
    comment id         -> source_id
    "youtube"          -> source
"""

import os

import requests

from core.record import Record, to_iso8601
from providers.youtube import config as yt_config

API_BASE = "https://www.googleapis.com/youtube/v3"
SEARCH_URL = f"{API_BASE}/search"
COMMENTS_URL = f"{API_BASE}/commentThreads"


class YouTubeError(RuntimeError):
    """Raised for API/quota errors so the caller can skip or stop gracefully."""


def _api_key() -> str:
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise YouTubeError(
            "Missing YOUTUBE_API_KEY environment variable. "
            "Set it in your shell (never commit the key)."
        )
    return key


def _get(url: str, params: dict) -> dict:
    """GET a YouTube API endpoint, raising YouTubeError with a concise reason."""
    params = dict(params, key=_api_key())
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        reason, message = "", resp.text[:200]
        try:
            err = resp.json().get("error", {})
            reason = (err.get("errors") or [{}])[0].get("reason", "")
            message = err.get("message", message)
        except ValueError:
            pass
        raise YouTubeError(f"{resp.status_code} {reason}: {message}")
    return resp.json()


def _search_videos(query, region_code, max_videos, published_after=None,
                   published_before=None):
    """Return a list of video ids for a search query, optionally date-bounded."""
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_videos, 50),
        "order": "relevance",
        "safeSearch": "none",
        "relevanceLanguage": "en",
    }
    if region_code:
        params["regionCode"] = region_code
    if published_after:
        params["publishedAfter"] = published_after
    if published_before:
        params["publishedBefore"] = published_before
    data = _get(SEARCH_URL, params)
    return [
        item["id"]["videoId"]
        for item in data.get("items", [])
        if item.get("id", {}).get("videoId")
    ]


def _video_comments(video_id, max_comments, order="relevance"):
    """Return raw commentThreads items for a video in the given order.

    `order` is "relevance" or "time" (the two orders YouTube supports). Pages
    through 100 at a time up to `max_comments`. Comments may be disabled or the
    video removed; in that case we skip the video rather than aborting the run.
    """
    items = []
    page_token = None
    while len(items) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(max_comments - len(items), 100),
            "order": order,
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token
        try:
            data = _get(COMMENTS_URL, params)
        except YouTubeError as exc:
            # Only note it once (the first order attempt) to avoid noise.
            if order == "relevance":
                print(f"  ! skipping video {video_id}: {exc}")
            break
        items.extend(data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return items


def _discover_videos(queries, region):
    """Sweep videos per year so comment dates spread across the timeline.

    Returns an ordered dict-like list of (video_id, year) with videos deduped,
    keeping the first year a video was found under.
    """
    found = {}  # video_id -> year
    n = len(queries)
    for offset, year in enumerate(yt_config.years()):
        after = f"{year}-01-01T00:00:00Z"
        before = f"{year + 1}-01-01T00:00:00Z"
        before_this = 0
        for k in range(yt_config.SEARCH_QUERIES_PER_YEAR):
            query = queries[(offset + k) % n]  # rotate queries across years
            try:
                ids = _search_videos(
                    query, region, yt_config.VIDEOS_PER_QUERY_PER_YEAR,
                    published_after=after, published_before=before,
                )
            except YouTubeError as exc:
                print(f"  ! search failed {query!r} {year}: {exc}")
                if "quota" in str(exc).lower():
                    print("  ! quota exhausted during search — stopping sweep.")
                    return found
                continue
            for vid in ids:
                if vid not in found:
                    found[vid] = year
                    before_this += 1
        print(f"  {year}: +{before_this} new videos")
    return found


def fetch(country, queries=None, max_results=None):
    """Pull YouTube comments for `country` (year-swept) and map to common records.

    Args:
        country: country name; must have queries configured unless `queries`
                 is passed explicitly.
        queries: optional override of the per-country search queries.
        max_results: optional overall cap on comments (defaults to config,
                     which is None = no cap, preserving the per-year spread).

    Returns a list of common-record dicts (no sentiment / relevance yet).
    """
    if queries is None:
        queries = yt_config.queries_for(country)
    if max_results is None:
        max_results = yt_config.MAX_RESULTS_DEFAULT  # may stay None
    region = yt_config.REGION_CODES.get(country)

    print(f"  discovering videos across {yt_config.YEAR_START}-{yt_config.YEAR_END} ...")
    videos = _discover_videos(queries, region)
    print(f"  collected {len(videos)} videos; pulling comments ...")

    records = []
    seen_comment_ids = set()

    def _capped():
        return max_results is not None and len(records) >= max_results

    for video_id in videos:
        if _capped():
            break
        for order in ("relevance", "time"):
            if _capped():
                break
            for item in _video_comments(
                video_id, yt_config.COMMENTS_PER_VIDEO_PER_ORDER, order
            ):
                if _capped():
                    break
                top = item.get("snippet", {}).get("topLevelComment", {})
                comment_id = top.get("id") or item.get("id")
                snippet = top.get("snippet", {})
                if not comment_id or comment_id in seen_comment_ids:
                    continue
                seen_comment_ids.add(comment_id)

                record = Record(
                    source="youtube",
                    source_id=comment_id,
                    country=country,
                    text=snippet.get("textDisplay", ""),
                    author=snippet.get("authorDisplayName", ""),
                    timestamp=to_iso8601(snippet.get("publishedAt")),
                    url=f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}",
                    engagement=int(snippet.get("likeCount", 0) or 0),
                )
                records.append(record.to_dict())

    return records
