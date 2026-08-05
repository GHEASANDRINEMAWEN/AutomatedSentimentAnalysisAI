"""Honesty rails — how much evidence a claim needs before we make it.

One source of truth for the "don't headline a finding on three comments"
thresholds, shared by the PDF report and the dashboard so the two can never
disagree about what counts as enough data.

The rails exist because the analysis is only as good as its thinnest cell. A
country with 7 records, or an aspect with 2 mentions, will happily produce a
100% positive score — and reporting that as a finding is worse than reporting
nothing. Every view that can slice the data thin should call `is_thin()` /
`has_enough()` and label or suppress accordingly.

No heavy imports here on purpose: dashboard.py imports this directly, while
core/report.py pulls in matplotlib and reportlab.
"""

# Below this many records a whole view (a country, a filtered slice, a segment)
# is flagged as indicative rather than reliable.
MIN_SAMPLE = 50

# Below this many mentions a single aspect cannot headline a finding, be ranked,
# or anchor a cross-country gap claim.
MIN_ASPECT_MENTIONS = 15

# Net sentiment (positive% - negative%) must differ by at least this much before
# a theme is called praised/criticised — under it the split is a coin flip.
NET_MARGIN = 5.0

# Two countries' net sentiment on the same aspect must differ by at least this
# much before the gap is worth stating. Wider than NET_MARGIN because a gap is a
# difference of two noisy numbers and carries both their errors.
GAP_MARGIN = 10.0

# Perception must move at least this many points across a period to count as a
# trend rather than drift.
TREND_MARGIN = 2.0


def is_thin(n) -> bool:
    """True when a view has too few records to read with confidence."""
    return int(n) < MIN_SAMPLE


def has_enough(mentions) -> bool:
    """True when an aspect has enough mentions to headline or be compared."""
    return int(mentions) >= MIN_ASPECT_MENTIONS


def thin_note(n, subject: str = "This view") -> str:
    """A one-line caveat for a thin view, or "" when the sample is adequate."""
    if not is_thin(n):
        return ""
    return (f"{subject} holds {int(n):,} record{'' if int(n) == 1 else 's'} — "
            f"below the {MIN_SAMPLE}-record threshold for a reliable read. "
            f"Treat the figures as indicative only.")
