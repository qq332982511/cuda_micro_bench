#!/usr/bin/env python3
"""Flatten micro_arch table rows into metric/value pairs."""
import json
from pathlib import Path

IN = Path(__file__).resolve().parent.parent / 'results' / 'micro_arch_metrics.json'
OUT = IN.with_name('micro_arch_metrics_flat.json')

def is_metric_row(cells):
    if len(cells) < 2:
        return False
    # Skip header-like rows if they are all uppercase/title words
    first = cells[0].strip()
    if not first or first.lower() in ('property','parameter','metric','component','case','sec','test type','anomaly type','behavior','artifact','mode','component','gpu','source','evidence'):
        return False
    return True

def main():
    docs = json.loads(IN.read_text(encoding='utf-8'))
    flat = []
    for doc in docs:
        for t in doc['tables']:
            caption = t['caption']
            for row in t['rows']:
                if is_metric_row(row):
                    metric = row[0].replace('\n',' ').strip()
                    value = row[1].replace('\n',' ').strip() if len(row)>1 else ''
                    extra = row[2].replace('\n',' ').strip() if len(row)>2 else ''
                    flat.append({
                        'file': doc['file'],
                        'caption': caption,
                        'metric': metric,
                        'value': value,
                        'extra': extra,
                    })
    OUT.write_text(json.dumps(flat, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote {len(flat)} metric rows to {OUT}')

if __name__ == '__main__':
    main()
