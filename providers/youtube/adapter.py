"""YouTube Data API v3 provider adapter.

Searches travel/tourism videos per country, pulls top-level comments via the
commentThreads endpoint, and maps each comment into the shared common record.
Pull + map only — no scoring, no filtering.

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


def _search_videos(query: str, region_code, max_videos: int):
    """Return a list of video ids for a search query."""
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_videos, 50),
        "order": "relevance",
        "safeSearch": "none",
    }
    if region_code:
        params["regionCode"] = region_code
    data = _get(SEARCH_URL, params)
    return [
        item["id"]["videoId"]
        for item in data.get("items", [])
        if item.get("id", {}).get("videoId")
    ]


def _video_comments(video_id: str, max_comments: int):
    """Return raw commentThreads items for a video, or [] if unavailable.

    Comments may be disabled or the video removed; in that case we skip the
    video rather than aborting the whole run.
    """
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": min(max_comments, 100),
        "order": "relevance",
        "textFormat": "plainText",
    }
    try:
        data = _get(COMMENTS_URL, params)
    except YouTubeError as exc:
        print(f"  ! skipping video {video_id}: {exc}")
        return []
    return data.get("items", [])


def fetch(country: str, queries=None, max_results=None):
    """Pull YouTube comments for `country` and map them to common records.

    Args:
        country: country name; must have queries configured in config.QUERIES
                 unless `queries` is passed explicitly.
        queries: optional override of the per-country search queries.
        max_results: overall cap on comments pulled (defaults to config value).

    Returns a list of common-record dicts (sentiment not yet attached).
    """
    if queries is None:
        queries = yt_config.queries_for(country)
    if max_results is None:
        max_results = yt_config.MAX_RESULTS_DEFAULT
    region = yt_config.REGION_CODES.get(country)

    records = []
    seen_comment_ids = set()

    for query in queries:
        if len(records) >= max_results:
            break
        try:
            video_ids = _search_videos(query, region, yt_config.MAX_VIDEOS_PER_QUERY)
        except YouTubeError as exc:
            print(f"  ! search failed for {query!r}: {exc}")
            # Quota exhausted: stop rather than hammering the API further.
            if "quota" in str(exc).lower():
                break
            continue

        print(f"  query {query!r}: {len(video_ids)} videos")
        for video_id in video_ids:
            if len(records) >= max_results:
                break
            for item in _video_comments(video_id, yt_config.MAX_COMMENTS_PER_VIDEO):
                if len(records) >= max_results:
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
