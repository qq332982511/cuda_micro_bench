#!/usr/bin/env python3
"""Validate that every metric path referenced by results/report.html exists in results/results.json."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
html = (ROOT / "results" / "report.html").read_text()
json_data = json.loads((ROOT / "results" / "results.json").read_text())

# Check tables, diagram bindings and comparison rows.
pattern = re.compile(r"\[((?:\s*'[^']+'\s*,)*\s*'[^']+'\s*)\]", re.MULTILINE)
missing = []
for m in pattern.findall(html):
    parts = [p.strip().strip("'") for p in m.split(",") if p.strip()]
    if not parts or parts[0] != "results":
        continue
    cur = json_data
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            missing.append(".".join(parts))
            break

if missing:
    print("Missing keys:")
    for k in sorted(set(missing)):
        print("  ", k)
    raise SystemExit(1)
print("All report metric keys are present in results.json.")

assert (ROOT / "docs" / "index.html").read_text() == html, "Published/local reports differ"
assert json.loads((ROOT / "docs" / "results.json").read_text()) == json_data, "Published/local data differ"

review = (ROOT / "REVIEW.md").read_text()
assert (ROOT / "results" / "review.md").read_text() == review
assert (ROOT / "docs" / "review.md").read_text() == review
