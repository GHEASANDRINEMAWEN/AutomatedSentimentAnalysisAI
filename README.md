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
  relevance.py      # mark(records): is each comment about travel/tourism?
  geo.py            # assign(records): which country is the text actually ABOUT?
  rails.py          # shared honesty thresholds — what counts as enough evidence
  segments.py       # tag(records): visitor segment (adventure/luxury/business/budget)
  sentiment.py      # score(records): transformer sentiment (roberta) — source-agnostic
  aspects.py        # tag(records): travel topics mentioned (rule-based, swappable)
  emotion.py        # tag(records): a single emotion cue (rule-based, swappable)
  store.py          # append/rewrite(records) to /data + CSV export, dedupe on source_id
  facts.py          # build(df): every figure the report is allowed to print
  narrative.py      # write(pack): Claude writes the analysis; numbers verified
  report.py         # build_pdf(df): intelligence-grade PDF — 7 sections + charts
run.py              # fetch -> relevance -> geo -> sentiment -> aspects -> emotion -> store
                    #   --reprocess           re-analyzes the stored data in place
                    #   --reassign-countries  re-derives attribution only (no model)
validate_report.py  # smoke-tests the PDF through the dashboard's own filters
validate_narrative.py  # proves the writing engine's guarantees (stubbed, offline)
config.py           # reads Reddit API credentials from environment variables
data/               # raw pulls as JSONL + CSV (git-ignored, keeps .gitkeep)
```

### Common record
Every provider maps into the same flat schema:

`source, source_id, country, country_query, country_source, text, author,
timestamp, url, engagement, rating, sentiment_label, sentiment_score, aspects,
emotion, relevance_kept`

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

### Analysis layer
Every record is enriched by source-agnostic analyzers in `core/`:
- **Sentiment** (`sentiment.py`) — the transformer model
  `cardiffnlp/twitter-roberta-base-sentiment-latest` sets `sentiment_label`
  (positive/neutral/negative) and a signed `sentiment_score` in [-1, 1]
  (`P(pos) - P(neg)`). The model downloads from the Hugging Face Hub on first
  use and caches locally — no API key. The `score(records)` interface is
  unchanged from the earlier VADER version.
- **Aspects** (`aspects.py`) — rule-based keyword tagging into travel topics
  (`safety, cost, scenery, food, wildlife, hospitality, transport`), stored as a
  comma-separated `aspects` column (blank if none).
- **Emotion** (`emotion.py`) — a single lightweight `emotion` label
  (`excited / disappointed / fearful / longing / neutral`) from keyword cues.
- **Segments** (`segments.py`) — a visitor-segment label in the `segment` column
  (`adventure / luxury / business / budget / unclassified`), scored by distinct
  keyword hits so overlapping vocabularies resolve sensibly ("we hiked to the
  ruins then stayed at a cheap hostel" is budget, not adventure). Roughly **9%
  of records carry a travel-style signal** — the rest are genuinely
  unclassifiable ("beautiful country, love the people"), and are left that way
  rather than guessed, which would poison every cohort average.

The aspect and emotion taggers are **swappable by design**: each exposes
`tag(records)` that fills its column, so an LLM-based tagger can replace the
rule-based one later without changing the CSV columns or any caller.

### Country attribution — what the record is ABOUT
A record arrives tagged with the country whose search query pulled it, and that
is only a hint: a "Nigeria travel" search surfaces videos that are really about
Ghana, and a comment under a Kenya vlog may be entirely about Zanzibar. Trusting
the query mislabels those rows across countries.

`core/geo.py` re-derives `country` from evidence, in three tiers recorded in
`country_source`:

| tier | how the country was decided |
|---|---|
| `content` | the record's own text names that country's places/demonyms |
| `context` | the text is place-less ("stunning!") but the rest of the SAME video's records point at one country |
| `query`   | no evidence anywhere — keep the search country |

`country_query` preserves the original pull country, so the pass is **idempotent**
and auditable. Strength is measured in *distinct* matched terms, ties go to the
pull country, and a record only moves when another country is strictly better
evidenced. Two guards keep it honest: viewer-origin mentions ("greetings from
Morocco" under a Tanzania video) are not evidence, and a video's pooled evidence
must be substantial, multi-record and dominant before it can re-home a whole
comment section.

Re-derive attribution over the stored data at any time — it needs no model and
takes seconds:
```bash
python run.py --reassign-countries    # attribution only
python run.py --retag                 # every rule-based tagger, incl. segments
```

### Honesty rails
`core/rails.py` holds one set of thresholds shared by the dashboard and the PDF
report, so the two can never disagree about what counts as enough evidence:

| rail | value | meaning |
|---|---|---|
| `MIN_SAMPLE` | 50 records | below this a view is flagged indicative, not reliable |
| `MIN_ASPECT_MENTIONS` | 15 mentions | below this an aspect cannot headline or be compared |
| `NET_MARGIN` | 5 points | below this a theme is an even split, not praised/criticised |
| `GAP_MARGIN` | 10 points | below this a cross-country gap is not worth stating |

Every view that can slice thin honours them: heatmap cells under the mention
floor render blank rather than as noise, thin cohorts are marked ⚠ and excluded
from rankings, and benchmark gaps are only computed where **both** sides clear
the floor.

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
pip install -r requirements-pipeline.txt   # full data-collection + analysis pipeline (run.py)
pip install -r requirements.txt            # dashboard only (streamlit/plotly/pandas)
```
`requirements.txt` is intentionally the lightweight dashboard set so it deploys
cleanly to Streamlit Community Cloud (which auto-installs `requirements.txt`).
The heavy model/collection deps (torch, transformers, praw, …) live in
`requirements-pipeline.txt`.

### Dashboard
```bash
python -m streamlit run dashboard.py
```
Reads `data/records.csv` and serves the filterable perception dashboard. It is
deployable as-is on Streamlit Community Cloud — point it at `dashboard.py`.

Tabs: **Overview**, **Compare** (cross-country, including the competitive
benchmark below), **Themes**, **Segments**, **Voices**, **Reviews**, **Data**.
The sidebar filters country, years, aspect, sentiment, visitor segment and data
source; every tab and the PDF report follow them.

#### Segments tab
Perception and per-aspect net sentiment broken down by traveller cohort — the
"luxury travellers rate hospitality X, business travellers rate Y" view. Cohorts
under `MIN_SAMPLE` are marked ⚠ and never ranked; aspect×segment cells under
`MIN_ASPECT_MENTIONS` render blank.

#### Competitive benchmark (Compare tab)
Pick any 2+ countries and a benchmark, and every rival is scored against it per
aspect: net-sentiment gap, complaint-share ratio, and a written finding such as
*"South Africa trails Zimbabwe on safety: −39 net vs +17 — a 56-point deficit
(135 vs 23 mentions). Complaint share is 87% higher."* Gaps are only computed
where both sides clear the mention floor, and thin countries raise a banner.

### PDF report
The sidebar's **Generate report** button renders the *current filtered view* —
country, year range, aspect, sentiment, source and segment — as an
intelligence-grade PDF in the Africa INSIGHTS house style: country and period on
a near-black title block, then seven numbered sections.

| § | Section | What it argues |
|---|---------|----------------|
| 01 | Month in Review | The verdict and the one thing to act on |
| 02 | Perception Overview | Score, net sentiment, breakdown *(Figure 1)* |
| 03 | Thematic Analysis | What drives the positive and the negative, with quoted voices *(Figure 2)* |
| 04 | Visitor Segments | How adventure / luxury / business / budget differ |
| 05 | Competitive Benchmarking | Standing against peer markets, overall and per theme |
| 06 | Trends & Signals | What is rising and falling *(Figures 3 & 4)* |
| 07 | Recommendations | Prescriptive actions, each tied to the finding that triggers it |

The report **prescribes rather than reports**: every finding implies an action —
*"safety is the top risk and worsening in Victoria Falls mentions → prioritise
visible security messaging"*, not *"safety sentiment is negative"*.

#### How the numbers stay honest
Three layers, and the separation is the point:

1. **`core/facts.py` computes.** Every figure — sentiment mix, per-aspect net,
   trend, segments, peer gaps, top voices, volumes — is derived from the filtered
   records into a JSON fact pack.
2. **`core/narrative.py` interprets.** The pack is handed to Claude
   (`claude-sonnet-4-6`, key read from `ANTHROPIC_API_KEY` in the environment and
   nowhere else) which writes the analysis *around* those figures. It is
   instructed never to calculate: no derived percentage, no summed total, no
   estimate. Where the analysis wants a figure the pack lacks, it writes a **DATA
   NOTE** instead.
3. **The code verifies.** Every numeral in the finished prose is checked back
   against the pack. A section citing an unverifiable figure is sent back once
   for repair; if it still fails, that section is replaced by the deterministic
   summary. Nothing unverified reaches the page.

**No key, no problem.** Without `ANTHROPIC_API_KEY` — or on a network failure, a
bad response, or a rate limit — the report still generates: the same seven
sections are written from the same fact pack by a deterministic template. The
methodology page always states which engine wrote the analysis.

Thin views say so rather than dressing up noise. The rails in `core/rails.py`
gate every claim, and each gap they open surfaces as a DATA NOTE callout: below
50 records the report flags itself indicative, no theme headlines a finding on
fewer than 15 mentions, and no cross-country gap is stated under 10 points.

`data/reports/africa-insights-zimbabwe-2015-2026.pdf` is a committed example.

Generate reports from the command line (same code path as the button):
```bash
python validate_report.py --country Ghana         # deterministic, free
python validate_report.py --country Zimbabwe --live   # writes with Claude
python validate_report.py --all-countries
python validate_report.py --country Kenya --aspects safety cost
```

Every run asserts that no figure in the prose is one the pipeline did not
compute. `python validate_narrative.py` tests the engine's guarantees —
verification, the repair pass, per-section substitution and every fallback path
— against a stubbed client, so it runs offline and costs nothing.

### Credentials — new developer setup
Copy the template and fill in **your own** keys. `.env` is git-ignored:
```bash
cp .env.example .env
```
`.env.example` lists every key the pipeline can use, where to get it, and what
it unlocks. Nothing is shared between developers, by design:

| Key | Needed for | Quota | Free? |
|---|---|---|---|
| `YOUTUBE_API_KEY` | `--provider youtube` (main path) | ~10,000 units/day **per key** | yes |
| `SERPAPI_KEY` | `--provider serpapi_reviews` | 250 searches/month **per account** | yes |
| `REDDIT_CLIENT_ID` / `_SECRET` / `_USER_AGENT` | `--provider reddit` | generous | yes |

**Every developer provisions their own key — do not share one.** Quota is
per-key, and a full country sweep costs ~3,600 of YouTube's ~10,000 daily units.
One shared key caps the whole team at 2–3 countries/day; four individual keys
give 8–12. Sharing costs throughput and gives up per-developer revocation.

Only the dashboard's deployment needs a shared secret, and it has none — it
reads the committed `data/records.csv` and calls no API.

### YouTube
Or set it in your shell instead of `.env` (never commit it):
```powershell
setx YOUTUBE_API_KEY "your-api-key"
```
The adapter reads the key only from `YOUTUBE_API_KEY`. Queries per country live
in `providers/youtube/config.py`. Configured markets: South Africa, Rwanda,
Kenya, Tanzania, Ghana, Nigeria, Zimbabwe, Senegal, Morocco, Egypt, Cameroon.

One country's full sweep costs ~3,600 quota units against the ~10,000/day
default, so plan on **two to three countries per day**.

### SerpApi reviews
```powershell
setx SERPAPI_KEY "your-key"
```
No review data has been collected yet — the place lists are configured for
South Africa, Nigeria, Ghana and Zimbabwe, but no run has had a key available.
Place lists per country live in `providers/serpapi_reviews/config.py`. The free
tier is 250 searches/month and the adapter hard-caps each run at
`MAX_API_CALLS_PER_RUN`. Without the key, `run.py --provider serpapi_reviews`
stops with a one-line message rather than a traceback.

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
python run.py --provider youtube --country "Nigeria"
python run.py --provider youtube --country "Ghana" --max-results 200
python run.py --provider serpapi_reviews --country "Zimbabwe"
python run.py --reprocess                                  # re-analyze everything
python run.py --reassign-countries                         # re-attribute only (fast)
```
This fetches, runs the analysis layer (relevance → country attribution →
sentiment → aspects → emotion), appends to `data/records.jsonl` (deduped on
source id), and prints a summary: number pulled, relevance pass rate, the date
range, how attribution was decided, the aspect breakdown, and sample rows.

## Adding a country
1. Add its tourism search queries to `QUERIES` in `providers/youtube/config.py`,
   plus a `REGION_CODES` entry (and `RELEVANCE_LANGUAGES` for non-English markets).
2. Add its place names to `PLACES` in `core/relevance.py` so its comments survive
   the relevance filter, and to `GAZETTEER` in `core/geo.py` so they are
   attributed to it. Keep both conservative — a term that also names something
   elsewhere (`kano` inside "volcano", `tamale` the dish) causes wrong rows.
3. Optionally add a handful of hotels/attractions to `PLACES` in
   `providers/serpapi_reviews/config.py`.
4. `python run.py --provider youtube --country "<Name>"`.

## Sources (in priority order)
1. **Reddit** — via the official API using [PRAW](https://praw.readthedocs.io/).
2. **YouTube** — comments via the [Data API v3](https://developers.google.com/youtube/v3).
3. **Google Trends** — relative search interest as a demand signal (planned).

## Status
Eleven African markets are collected and scored (35.5k records), served by a
filterable Streamlit dashboard with a one-click intelligence-grade PDF report:
seven numbered sections written by Claude around figures the pipeline computes,
with every published number verified back against them. Every record is
attributed to the country its text is about, not the query that found it.
Google Trends is the next signal.
