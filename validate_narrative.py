"""Prove the narrative engine's guarantees without spending an API call.

The report's central claim is that a number on the page was computed from the
records — never invented by the language model. That claim rests on three
mechanisms, and this script tests all three against a real fact pack built from
data/records.csv:

  1. **Verification.** Every numeral in the prose is checked against
     `facts.allowed_numbers()`. A figure that is not in the pack is caught.
  2. **Repair, then substitution.** A draft citing an unverifiable figure is sent
     back once; if it comes back wrong again, that section is replaced by the
     deterministic template rather than published.
  3. **It always produces a report.** Missing key, network failure, malformed
     response — every path lands on the template.

Claude is stubbed with a fake client that returns canned payloads, so the tests
run offline, deterministically and for free. What is NOT tested here is whether
the live API accepts the request; `validate_report.py --live` does that when a
key is present.

Usage:
    python validate_narrative.py
    python validate_narrative.py --country Kenya
"""

import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import dashboard
from core import facts, narrative

PASS, FAIL = "  PASS", "  FAIL"
_results = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    _results.append(bool(condition))
    print(f"{PASS if condition else FAIL}  {name}" + (f" — {detail}" if detail else ""))
    return bool(condition)


# --------------------------------------------------------------------------- #
# A fake Anthropic client
# --------------------------------------------------------------------------- #
class FakeBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeResponse:
    def __init__(self, payload):
        self.content = [FakeBlock(type="tool_use", name="submit_report",
                                  id="toolu_fake", input=payload)]


class FakeMessages:
    def __init__(self, client):
        self.client = client

    def create(self, **kwargs):
        self.client.calls.append(kwargs)
        if self.client.raises:
            raise RuntimeError("simulated API failure")
        index = min(len(self.client.calls) - 1, len(self.client.payloads) - 1)
        return FakeResponse(self.client.payloads[index])


class FakeClient:
    """Records every request and replays a scripted list of payloads."""

    def __init__(self, payloads, raises: bool = False):
        self.payloads = payloads
        self.raises = raises
        self.calls = []
        self.messages = FakeMessages(self)


def payload(prose_by_id: dict, headline: str = "A test headline.") -> dict:
    return {
        "headline": headline,
        "sections": [{"id": sid, "prose": prose_by_id.get(sid, "Placeholder prose."),
                      "data_notes": []}
                     for sid in narrative.SECTION_IDS],
    }


def with_fake(monkey_client):
    """Swap the engine's client factory for the duration of one call."""
    original = narrative._client
    narrative._client = lambda: (monkey_client, "")
    return original


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_allowlist(pack):
    allowed = facts.allowed_numbers(pack)
    sent = pack["sentiment"]
    ok = True
    for label, value in (("perception score", sent["perception_score"]),
                         ("positive %", sent["positive_pct"]),
                         ("record count", pack["volume"]["records"])):
        hit = any(abs(float(value) - a) <= narrative.ROUNDING_TOLERANCE
                  for a in allowed)
        ok &= check(f"allowlist contains the {label}", hit, f"{value}")
    # A figure nobody computed must not be present.
    invented = 987654.0
    ok &= check("allowlist rejects an invented figure",
                not any(abs(invented - a) <= narrative.ROUNDING_TOLERANCE
                        for a in allowed), f"{invented:g}")
    # Rounded restatements of a computed figure must pass.
    rounded = round(sent["positive_pct"])
    ok &= check("allowlist accepts a rounded restatement",
                any(abs(rounded - a) <= narrative.ROUNDING_TOLERANCE
                    for a in allowed), f"{sent['positive_pct']} -> {rounded}")
    return ok


def test_detection(pack):
    allowed = facts.allowed_numbers(pack)
    n = pack["volume"]["records"]
    clean = f"The view holds {n:,} records."
    dirty = "Perception rose to 91.4 out of 100, up from 88.2 last quarter."
    ok = check("clean prose passes verification",
               not narrative.unverified(clean, allowed))
    bad = narrative.unverified(dirty, allowed)
    ok &= check("invented figures are caught", len(bad) >= 2,
                f"flagged {', '.join(f'{b:g}' for b in bad)}")
    ok &= check("bold tags do not hide a bad figure",
                bool(narrative.unverified("<b>91.4</b> out of 100", allowed)))
    return ok


def test_template_self_consistency(pack):
    """The deterministic fallback must obey its own rule.

    This is the strongest check in the file: it proves the prose that ships when
    no model is available cites nothing the fact pack did not compute.
    """
    sections, headline = narrative.template_sections(pack)
    problems = narrative.verify(sections, pack)
    failed = {k: v for k, v in problems.items() if v}
    detail = "; ".join(f"{k}: {', '.join(f'{n:g}' for n in v)}"
                       for k, v in failed.items())
    ok = check("every figure in the template narrative is computed", not failed,
               detail or f"{len(sections)} sections clean")
    ok &= check("the headline cites only computed figures",
                not narrative.unverified(headline, facts.allowed_numbers(pack)),
                headline)
    return ok


def test_clean_model_draft(pack):
    n = pack["volume"]["records"]
    good = {sid: f"This view rests on {n:,} records." for sid in narrative.SECTION_IDS}
    client = FakeClient([payload(good, "Model headline.")])
    original = with_fake(client)
    try:
        result = narrative.write(pack)
    finally:
        narrative._client = original
    ok = check("a clean draft is published as written", result.engine == "claude",
               f"engine={result.engine}")
    ok &= check("one API call for a clean draft", len(client.calls) == 1,
                f"{len(client.calls)} call(s)")
    ok &= check("the model's headline is used",
                result.headline == "Model headline.", result.headline)
    ok &= check("all seven sections are returned", len(result.ordered()) == 7,
                f"{len(result.ordered())} sections")
    return ok


def test_repair_then_publish(pack):
    n = pack["volume"]["records"]
    bad = {sid: "Perception reached 91.4 out of 100." for sid in narrative.SECTION_IDS}
    good = {sid: f"This view rests on {n:,} records." for sid in narrative.SECTION_IDS}
    client = FakeClient([payload(bad), payload(good)])
    original = with_fake(client)
    try:
        result = narrative.write(pack)
    finally:
        narrative._client = original
    ok = check("a bad draft triggers exactly one repair pass",
               len(client.calls) == 2, f"{len(client.calls)} call(s)")
    ok &= check("the repaired draft is published", result.engine == "claude",
                f"engine={result.engine}")
    repair = client.calls[1]["messages"][-1]["content"]
    body = repair[0]["content"] if isinstance(repair, list) else repair
    ok &= check("the repair request names the offending figure",
                "91.4" in body)
    ok &= check("the repair is returned as a tool_result",
                isinstance(repair, list) and repair[0]["type"] == "tool_result")
    return ok


def test_unrepairable_falls_back(pack):
    bad = {sid: "Perception reached 91.4 out of 100." for sid in narrative.SECTION_IDS}
    client = FakeClient([payload(bad), payload(bad)])
    original = with_fake(client)
    try:
        result = narrative.write(pack)
    finally:
        narrative._client = original
    ok = check("prose that cannot be repaired is not published",
               result.engine == "claude+template", f"engine={result.engine}")
    ok &= check("no unverifiable figure survives into the report",
                not any(v for v in narrative.verify(result.sections, pack).values()))
    ok &= check("the report still has all seven sections",
                len(result.ordered()) == 7, f"{len(result.ordered())} sections")
    return ok


def test_partial_substitution(pack):
    """A bad section is swapped out; its clean neighbours are kept."""
    n = pack["volume"]["records"]
    mixed = {sid: f"This view rests on {n:,} records."
             for sid in narrative.SECTION_IDS}
    mixed["recommendations"] = "Spend 91.4 million on the campaign."
    client = FakeClient([payload(mixed), payload(mixed)])
    original = with_fake(client)
    try:
        result = narrative.write(pack)
    finally:
        narrative._client = original
    kept = result.sections["month_in_review"]["prose"]
    ok = check("a clean section survives its neighbour's failure",
               f"{n:,}" in kept and "91.4" not in kept)
    ok &= check("the failing section is replaced",
                "91.4" not in result.sections["recommendations"]["prose"])
    return ok


def test_failure_paths(pack):
    client = FakeClient([], raises=True)
    original = with_fake(client)
    try:
        result = narrative.write(pack)
    finally:
        narrative._client = original
    ok = check("an API failure falls back instead of raising",
               result.engine == "template", f"engine={result.engine}")
    ok &= check("the failure is recorded in the log",
                any("failed" in line for line in result.log),
                result.log[-1] if result.log else "")

    client = FakeClient([{"headline": "x", "sections": []}])
    original = with_fake(client)
    try:
        result = narrative.write(pack)
    finally:
        narrative._client = original
    ok &= check("an empty response falls back", result.engine == "template")

    original = narrative._client
    narrative._client = lambda: (None, "ANTHROPIC_API_KEY is not set")
    try:
        result = narrative.write(pack)
    finally:
        narrative._client = original
    ok &= check("a missing key falls back", result.engine == "template")
    ok &= check("the missing key is explained in the log",
                any("ANTHROPIC_API_KEY" in line for line in result.log))
    return ok


def test_request_shape(pack):
    """The request we would send is the one the model contract requires."""
    n = pack["volume"]["records"]
    good = {sid: f"This view rests on {n:,} records." for sid in narrative.SECTION_IDS}
    client = FakeClient([payload(good)])
    original = with_fake(client)
    try:
        narrative.write(pack)
    finally:
        narrative._client = original
    call = client.calls[0]
    ok = check("model is claude-sonnet-4-6", call["model"] == "claude-sonnet-4-6",
               call["model"])
    ok &= check("adaptive thinking is requested",
                call.get("thinking") == {"type": "adaptive"})
    ok &= check("budget_tokens is not sent (removed on 4.6+)",
                "budget_tokens" not in str(call.get("thinking")))
    ok &= check("no sampling parameters are sent",
                not {"temperature", "top_p", "top_k"} & set(call))
    ok &= check("the submit_report tool is offered",
                call["tools"][0]["name"] == "submit_report")
    ok &= check("no assistant prefill (400s on 4.6+)",
                call["messages"][-1]["role"] == "user")
    ok &= check("the fact pack is in the prompt",
                "perception_score" in call["messages"][0]["content"])
    ok &= check("the system prompt forbids calculation",
                "You do not calculate" in call["system"])
    ok &= check("no API key appears in the request",
                "sk-ant" not in str(call))
    return ok


def test_no_hardcoded_key():
    """The key must come from the environment and nowhere else."""
    import pathlib
    ok = True
    for path in pathlib.Path("core").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        ok &= check(f"no literal key in {path.as_posix()}", "sk-ant-" not in source)
    source = pathlib.Path("core/narrative.py").read_text(encoding="utf-8")
    ok &= check("the key is read from os.environ",
                'os.environ.get("ANTHROPIC_API_KEY")' in source)
    return ok


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="Zimbabwe")
    parser.add_argument("--years", type=int, nargs=2, default=(2015, 2026))
    args = parser.parse_args()

    df = dashboard.load_data()
    sources = sorted(df["source"].dropna().unique().tolist())
    filtered = dashboard.filter_data(
        df, country=args.country, year_range=tuple(args.years), aspects=[],
        sentiments=list(dashboard.SENTIMENTS), sources=sources, kept_only=True)
    benchmark = dashboard.filter_data(
        df, country="All", year_range=tuple(args.years), aspects=[],
        sentiments=list(dashboard.SENTIMENTS), sources=sources, kept_only=True)
    pack = facts.build(filtered, country=args.country, granularity="Year",
                       benchmark_df=benchmark)
    print(f"Fact pack: {args.country}, {pack['volume']['records']:,} records, "
          f"{len(facts.allowed_numbers(pack)):,} verified figures\n")

    for title, test in (
        ("Number allowlist", lambda: test_allowlist(pack)),
        ("Invented-figure detection", lambda: test_detection(pack)),
        ("Template self-consistency", lambda: test_template_self_consistency(pack)),
        ("Clean model draft", lambda: test_clean_model_draft(pack)),
        ("Repair pass", lambda: test_repair_then_publish(pack)),
        ("Unrepairable draft", lambda: test_unrepairable_falls_back(pack)),
        ("Partial substitution", lambda: test_partial_substitution(pack)),
        ("Failure paths", lambda: test_failure_paths(pack)),
        ("Request shape", lambda: test_request_shape(pack)),
        ("No hard-coded credentials", test_no_hardcoded_key),
    ):
        print(f"{title}")
        test()
        print()

    passed, total = sum(_results), len(_results)
    print(f"{passed}/{total} checks passed.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
