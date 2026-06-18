# AutomatedSentimentAnalysisAI

**Real-time tourism sentiment across free and open sources, per country.**

This project measures how people feel about travelling to a given country by
collecting public posts and comments from free/open data sources and scoring
their sentiment. The goal is a live, per-country read on tourism sentiment that
costs nothing to run.

## Sources (in priority order)
1. **Reddit** (first) — via the official API using [PRAW](https://praw.readthedocs.io/).
2. **YouTube** — comments and video metadata (planned).
3. **Google Trends** — relative search interest as a demand signal (planned).

## How it works
1. **Collect** — pull posts/comments mentioning a country's tourism (Reddit first).
2. **Score** — run each item through a sentiment model. We start with
   [vaderSentiment](https://github.com/cjhutto/vaderSentiment), a lightweight,
   rule-based scorer suited to short social text.
3. **Aggregate** — roll scores up to a per-country sentiment over time.
4. **Store** — raw pulls are saved under `/data` (git-ignored).

## Setup
```bash
pip install -r requirements.txt
```

Set your Reddit API credentials as environment variables (never commit them):
```powershell
setx REDDIT_CLIENT_ID     "your-client-id"
setx REDDIT_CLIENT_SECRET "your-client-secret"
setx REDDIT_USER_AGENT    "tourism-sentiment by u/your-username"
```
`config.py` reads these from the environment. You may instead put them in a
local `.env` file, which is git-ignored.

## Project layout
```
config.py         # reads API keys from environment variables
requirements.txt  # praw, requests, vaderSentiment
data/             # raw pulls (git-ignored, keeps .gitkeep)
```

## Status
Early setup. Reddit collection and VADER scoring come first; YouTube and
Google Trends follow.
