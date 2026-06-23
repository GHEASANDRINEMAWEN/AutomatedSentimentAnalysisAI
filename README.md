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
    adapter.py      # fetch(country, queries, max_results) -> common records
    config.py       # per-country search queries + quota caps
  reddit/
    adapter.py      # Reddit fetch (moved from collect_reddit.py)
    config.py       # subreddits per country
core/
  record.py         # the shared common record + timestamp normalization
  sentiment.py      # score(records) with VADER — source-agnostic
  store.py          # append(records) to /data, dedupe on source_id
run.py              # orchestrates: pick provider + country -> fetch -> score -> store
config.py           # reads Reddit API credentials from environment variables
data/               # raw pulls as JSON Lines (git-ignored, keeps .gitkeep)
```

### Common record
Every provider maps into the same flat schema:

`source, source_id, country, text, author, timestamp, url, engagement,
sentiment_label, sentiment_score`

Two fields are mandatory for traceability: **`timestamp`** (the item's publish
date, ISO 8601 UTC) and **`url`** (a direct link to the comment/post). These let
the dashboard trace every row back to its source and filter/group by date
(e.g. 2015 to now). Dates are captured per item — we never loop over years.

### Adapter rules
- An adapter pulls from its platform and maps into the common record. That's it
  — **no scoring, no filtering**.
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
