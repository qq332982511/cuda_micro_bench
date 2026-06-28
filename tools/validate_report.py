#!/usr/bin/env python3
"""Validate that every metric path referenced by results/report.html exists in results/results.json."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
html = (ROOT / "results" / "report.html").read_text()
json_data = json.loads((ROOT / "results" / "results.json").read_text())

# Extract key arrays from SECTIONS map.
pattern = re.compile(r"key:\s*\[([^\]]+)\]", re.MULTILINE)
missing = []
for m in pattern.findall(html):
    parts = [p.strip().strip("'") for p in m.split(",") if p.strip()]
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
