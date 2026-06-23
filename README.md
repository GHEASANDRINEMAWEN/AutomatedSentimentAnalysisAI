# AutomatedSentimentAnalysisAI

**Real-time tourism sentiment across free and open sources, per country.**

This project measures how people feel about travelling to a given country by
collecting public posts and comments from free/open data sources and scoring
their sentiment. The goal is a live, per-country read on tourism sentiment that
costs nothing to run.

## Architecture

Each data provider lives in its own folder and only knows how to pull from its
platform and map results into one **common record**. Scoring and storage are
source-agnostic and shared.

```
providers/
  youtube/
    adapter.py      # comments + transcripts -> common records
    config.py       # per-country search queries, year span, quota/transcript caps
  reddit/
    adapter.py      # Reddit fetch (moved from collect_reddit.py)
    config.py       # subreddits per country
core/
  record.py         # the shared common record + timestamp normalization
  sentiment.py      # score(records) with VADER — source-agnostic
  relevance.py      # mark(records): is each comment about travel/tourism?
  store.py          # append(records) to /data + CSV export, dedupe on source_id
run.py              # orchestrates: fetch -> mark relevance -> score -> store -> report
config.py           # reads Reddit API credentials from environment variables
data/               # raw pulls as JSONL + CSV (git-ignored, keeps .gitkeep)
```

### Common record
Every provider maps into the same flat schema:

`source, source_id, country, text, author, timestamp, url, engagement,
sentiment_label, sentiment_score, relevance_kept`

Two fields are mandatory for traceability: **`timestamp`** (the item's publish
date, ISO 8601 UTC) and **`url`** (a direct link to the comment/post). These let
the dashboard trace every row back to its source and filter/group by date
(e.g. 2015 to now). Dates are captured per item — we never loop over comment years.

### Date coverage (2015–2026)
Comments can't be filtered by year, but a video's comments cluster around when it
was posted. So the YouTube adapter sweeps travel videos published in **each year**
(via the API's `publishedAfter`/`publishedBefore` filters) and pulls each video's
comments in both `relevance` and `time` order. The result is a comment-date spread
across the whole 2015–2026 range rather than bunched in recent years. The year
span lives in `providers/youtube/config.py`.

### Two YouTube signals
The YouTube provider produces two kinds of records, both in the common schema
and distinguishable by `source`:
- **`youtube`** — top-level video comments (via the Data API).
- **`youtube_transcript`** — each collected video's transcript (via
  [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/),
  no API key), split into ~sentence-sized chunks. The chunk's `timestamp` is the
  video's publish date, `url` the video link, and `author` the channel name.
  Videos without captions are skipped. Transcript chunks go through the same
  relevance filter and sentiment scorer as comments.

### Relevance filter
`core/relevance.py` marks each record with **`relevance_kept`** (True/False)
based on whether the text looks like tourism/destination talk (keywords such as
*visit, travel, safari, safe, beautiful, holiday* and place names like *Cape Town,
Kruger, Garden Route*) versus creator-directed chatter, one-word reactions, or
pure emoji. Nothing is deleted — filtered-out rows stay in the store/CSV so the
filter can be reviewed.

### Adapter rules
- An adapter pulls from its platform and maps into the common record. That's it
  — **no scoring, no filtering** (relevance and scoring happen in `core/`).
- Every record must populate `timestamp` and `url`.
- `country` is a parameter; queries/subreddits are defined per country in each
  provider's `config.py`, so adding a country is just adding an entry.

## Setup
```bash
pip install -r requirements.txt
```

### YouTube
Set your YouTube Data API v3 key as an environment variable (never commit it):
```powershell
setx YOUTUBE_API_KEY "your-api-key"
```
The adapter reads the key only from `YOUTUBE_API_KEY`. Queries per country live
in `providers/youtube/config.py` (South Africa is configured to start).

### Reddit
```powershell
setx REDDIT_CLIENT_ID     "your-client-id"
setx REDDIT_CLIENT_SECRET "your-client-secret"
setx REDDIT_USER_AGENT    "tourism-sentiment by u/your-username"
```
`config.py` reads these from the environment. You may instead put any secret in
a local `.env` file, which is git-ignored.

## Usage
```bash
python run.py                                              # youtube, South Africa
python run.py --provider youtube --country "South Africa" --max-results 200
python run.py --provider reddit  --country "South Africa"
```
This fetches, scores with VADER, appends to `data/records.jsonl` (deduped on
source id), and prints a summary: number pulled, the date range, and sample rows.

## Sources (in priority order)
1. **Reddit** — via the official API using [PRAW](https://praw.readthedocs.io/).
2. **YouTube** — comments via the [Data API v3](https://developers.google.com/youtube/v3).
3. **Google Trends** — relative search interest as a demand signal (planned).

## Status
Reddit and YouTube providers feed a shared VADER scorer and JSONL store.
Google Trends and a dashboard come next.
