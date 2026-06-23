"""YouTube Data API v3 provider adapter.

Two signals, both mapped into the shared common record (pull + map only; no
scoring, no filtering — relevance and scoring happen in core/):

  * comments       (source="youtube")            via the commentThreads endpoint
  * transcripts    (source="youtube_transcript") via youtube-transcript-api

To spread comment dates across the whole timeline, videos are searched per year
using the API's publishedAfter/publishedBefore filters; a video's comments
cluster around when it was posted. Each video's comments are pulled in both
"relevance" and "time" order so one popular video can't dominate.

The Data API key is read only from the YOUTUBE_API_KEY environment variable and
is never hard-coded. Transcripts need no key.

Field mapping (comment -> common record):
    textDisplay        -> text
    authorDisplayName  -> author
    likeCount          -> engagement
    publishedAt        -> timestamp   (the comment's publish date)
    watch?v=..&lc=..   -> url         (direct link to the comment)
    comment id         -> source_id
    "youtube"          -> source

Field mapping (transcript chunk -> common record):
    chunk of transcript text   -> text
    video's channel title      -> author
    video's publishedAt        -> timestamp   (the video's publish date)
    watch?v=..                 -> url          (the video link)
    "<video_id>-t<index>"      -> source_id
    "youtube_transcript"       -> source
"""

import os

import requests

from core.record import Record, to_iso8601
from providers.youtube import config as yt_config

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

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
    """Return [(video_id, snippet), ...] for a search query, optionally dated."""
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
    out = []
    for item in data.get("items", []):
        vid = item.get("id", {}).get("videoId")
        if vid:
            out.append((vid, item.get("snippet", {})))
    return out


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

    Returns an ordered dict: video_id -> {year, channel, published_at}, videos
    deduped (first year a video was found under is kept).
    """
    found = {}
    n = len(queries)
    for offset, year in enumerate(yt_config.years()):
        after = f"{year}-01-01T00:00:00Z"
        before = f"{year + 1}-01-01T00:00:00Z"
        new_this_year = 0
        for k in range(yt_config.SEARCH_QUERIES_PER_YEAR):
            query = queries[(offset + k) % n]  # rotate queries across years
            try:
                items = _search_videos(
                    query, region, yt_config.VIDEOS_PER_QUERY_PER_YEAR,
                    published_after=after, published_before=before,
                )
            except YouTubeError as exc:
                print(f"  ! search failed {query!r} {year}: {exc}")
                if "quota" in str(exc).lower():
                    print("  ! quota exhausted during search — stopping sweep.")
                    return found
                continue
            for vid, snippet in items:
                if vid not in found:
                    found[vid] = {
                        "year": year,
                        "channel": snippet.get("channelTitle", ""),
                        "published_at": snippet.get("publishedAt", ""),
                    }
                    new_this_year += 1
        print(f"  {year}: +{new_this_year} new videos")
    return found


def _fetch_transcript(video_id):
    """Return a list of transcript segment texts for a video, or [] if none.

    Videos without captions (disabled/unavailable) are handled gracefully by
    returning an empty list.
    """
    if YouTubeTranscriptApi is None:
        return []
    try:
        fetched = YouTubeTranscriptApi().fetch(
            video_id, languages=list(yt_config.TRANSCRIPT_LANGUAGES)
        )
        return [getattr(seg, "text", "") for seg in fetched]
    except Exception:
        # NoTranscriptFound / TranscriptsDisabled / network, etc. -> skip video.
        return []


def _chunk_segments(texts, max_chars):
    """Group short transcript segments into ~max_chars chunks (a few sentences)."""
    chunks = []
    buffer = ""
    for text in texts:
        piece = " ".join((text or "").split())
        if not piece:
            continue
        if buffer and len(buffer) + 1 + len(piece) > max_chars:
            chunks.append(buffer)
            buffer = piece
        else:
            buffer = f"{buffer} {piece}".strip()
    if buffer:
        chunks.append(buffer)
    return chunks


def _collect_transcripts(country, videos):
    """Fetch + chunk transcripts for the discovered videos into common records."""
    records = []
    videos_with_transcript = 0
    for video_id, meta in videos.items():
        segments = _fetch_transcript(video_id)
        if not segments:
            continue
        videos_with_transcript += 1
        chunks = _chunk_segments(segments, yt_config.TRANSCRIPT_CHUNK_CHARS)
        chunks = chunks[: yt_config.MAX_TRANSCRIPT_CHUNKS_PER_VIDEO]
        url = f"https://www.youtube.com/watch?v={video_id}"
        for index, chunk in enumerate(chunks):
            record = Record(
                source="youtube_transcript",
                source_id=f"{video_id}-t{index}",
                country=country,
                text=chunk,
                author=meta.get("channel", ""),
                timestamp=to_iso8601(meta.get("published_at")),
                url=url,
                engagement=0,
            )
            records.append(record.to_dict())
    print(f"  transcripts: {videos_with_transcript}/{len(videos)} videos had "
          f"transcripts -> {len(records)} chunks")
    return records


def fetch(country, queries=None, max_results=None):
    """Pull YouTube comments + transcript chunks for `country` as common records.

    Args:
        country: country name; must have queries configured unless `queries`
                 is passed explicitly.
        queries: optional override of the per-country search queries.
        max_results: optional overall cap on COMMENTS (defaults to config,
                     which is None = no cap, preserving the per-year spread).
                     Transcript volume is bounded by its own config caps.

    Returns a list of common-record dicts (no sentiment / relevance yet),
    mixing source="youtube" (comments) and source="youtube_transcript".
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

    if yt_config.INCLUDE_TRANSCRIPTS:
        print("  fetching transcripts (no API key needed) ...")
        records.extend(_collect_transcripts(country, videos))

    return records
