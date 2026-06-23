"""Append common records to /data, deduping on source id.

Records are stored as JSON Lines (one JSON object per line) in a single file,
which makes appending cheap and dedupe straightforward. The dedupe key is
`source:source_id` so ids are unique within a platform and never collide
across platforms.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STORE_FILE = DATA_DIR / "records.jsonl"


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
