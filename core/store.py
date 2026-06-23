"""Append common records to /data, deduping on source id.

Records are stored as JSON Lines (one JSON object per line) in a single file,
which makes appending cheap and dedupe straightforward. The dedupe key is
`source:source_id` so ids are unique within a platform and never collide
across platforms.

A spreadsheet-friendly CSV mirror (records.csv) can be regenerated from the
JSONL store via `export_csv()`.
"""

import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STORE_FILE = DATA_DIR / "records.jsonl"
CSV_FILE = DATA_DIR / "records.csv"

# Column order for the CSV export — one record per row.
CSV_COLUMNS = (
    "source",
    "country",
    "timestamp",
    "author",
    "text",
    "sentiment_label",
    "sentiment_score",
    "engagement",
    "url",
)


def _key(rec: dict) -> str:
    return f'{rec.get("source")}:{rec.get("source_id")}'


def existing_keys() -> set:
    """Dedupe keys already present in the store."""
    keys = set()
    if STORE_FILE.exists():
        for line in STORE_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(_key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return keys


def append(records) -> int:
    """Append new records to the store, skipping any already-seen source ids.

    Dedupes both against what's already on disk and within this batch.
    Returns the number of records actually written.
    """
    DATA_DIR.mkdir(exist_ok=True)
    seen = existing_keys()
    added = 0
    with STORE_FILE.open("a", encoding="utf-8") as fh:
        for rec in records:
            key = _key(rec)
            if key in seen:
                continue
            seen.add(key)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            added += 1
    return added


def read_all() -> list:
    """Load every record from the JSONL store as a list of dicts."""
    records = []
    if STORE_FILE.exists():
        for line in STORE_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def export_csv(records=None) -> int:
    """Write the store to data/records.csv, one record per row.

    Regenerates the whole CSV from the JSONL store (or from `records` if given)
    so the table always mirrors what's stored. Uses UTF-8 with a BOM so Excel
    opens emoji/accented text correctly. Returns the number of rows written.
    """
    if records is None:
        records = read_all()
    DATA_DIR.mkdir(exist_ok=True)
    with CSV_FILE.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {col: rec.get(col, "") for col in CSV_COLUMNS}
            # Flatten newlines so each record stays on a single CSV row.
            text = row.get("text") or ""
            row["text"] = " ".join(str(text).split())
            writer.writerow(row)
    return len(records)
