#!/usr/bin/env python3
"""Extract tables and key-value lists from zartbot/micro_arch HTML docs."""
import json
import re
from html.parser import HTMLParser
from pathlib import Path

# Default: clone zartbot/micro_arch into the repo root or pass your own path.
REF_DIR = Path('micro_arch')
OUT = Path(__file__).resolve().parent.parent / 'results' / 'micro_arch_metrics.json'

class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self._stack = []
        self._cur = None
        self._cell = None
        self._text = []
        self._caption = []
        self._in_caption = False
        self._h2 = None
        self._h3 = None
        self._h4 = None

    def handle_starttag(self, tag, attrs):
        self._stack.append(tag)
        if tag == 'table':
            self._cur = {'rows': []}
            self._caption = []
        elif tag == 'caption':
            self._in_caption = True
        elif tag in ('th','td'):
            self._cell = []
        elif tag in ('h2','h3','h4'):
            self._text = []

    def handle_endtag(self, tag):
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
        if tag in ('th','td') and self._cell is not None:
            text = ' '.join(self._cell).strip()
            if self._cur is not None:
                if not self._cur['rows'] or tag == 'tr' or (self._stack and self._stack[-1] == 'tr'):
                    # new row if needed
                    if not self._cur['rows'] or self._cur['rows'][-1]['closed']:
                        self._cur['rows'].append({'closed': False, 'cells': []})
                self._cur['rows'][-1]['cells'].append(text)
                self._cur['rows'][-1]['closed'] = False
            self._cell = None
        elif tag == 'tr' and self._cur is not None:
            if self._cur['rows']:
                self._cur['rows'][-1]['closed'] = True
        elif tag == 'caption':
            self._in_caption = False
            if self._cur is not None:
                self._cur['caption'] = ' '.join(self._caption).strip()
        elif tag == 'table' and self._cur is not None:
            self._cur['caption'] = self._cur.get('caption') or self._make_context()
            self.tables.append(self._cur)
            self._cur = None
        elif tag in ('h2','h3','h4'):
            text = ' '.join(self._text).strip()
            if tag == 'h2':
                self._h2 = text
                self._h3 = None
                self._h4 = None
            elif tag == 'h3':
                self._h3 = text
                self._h4 = None
            elif tag == 'h4':
                self._h4 = text

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._in_caption:
            self._caption.append(text)
        elif self._cell is not None:
            self._cell.append(text)
        elif self._stack and self._stack[-1] in ('h2','h3','h4'):
            self._text.append(text)

    def _make_context(self):
        parts = [self._h2, self._h3, self._h4]
        return ' > '.join(p for p in parts if p)

def main():
    docs = []
    for html in sorted(REF_DIR.rglob('*.html')):
        rel = html.relative_to(REF_DIR)
        parser = TableParser()
        parser.feed(html.read_text(encoding='utf-8', errors='ignore'))
        # Clean rows
        clean_tables = []
        for t in parser.tables:
            rows = [r['cells'] for r in t['rows'] if r['cells']]
            if rows:
                clean_tables.append({'caption': t.get('caption',''), 'rows': rows})
        docs.append({'file': str(rel), 'tables': clean_tables})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote {len(docs)} docs with tables to {OUT}')

if __name__ == '__main__':
    main()
