from __future__ import annotations
from pathlib import Path
import ast
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    ROOT/'core'/'collector.py',
    ROOT/'core'/'schema.py',
    ROOT/'scripts'/'collect_sequence.py',
    ROOT/'scripts'/'import_video_sequence.py',
    ROOT/'scripts'/'inspect_dataset.py',
    ROOT/'scripts'/'validate_phase1_dataset.py',
    ROOT/'configs'/'default.yaml',
    ROOT/'configs'/'dataset.yaml',
]

def main():
    errors=[]
    for p in REQUIRED:
        if not p.is_file():
            errors.append(f'MISSING: {p.relative_to(ROOT)}')
            continue
        if p.suffix=='.py':
            if p.stat().st_size == 0:
                errors.append(f'EMPTY: {p.relative_to(ROOT)}')
                continue
            try:
                ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
            except SyntaxError as e:
                errors.append(f'SYNTAX: {p.relative_to(ROOT)}: {e}')
    if errors:
        print('PHASE 1 SOURCE AUDIT: FAIL')
        for e in errors: print(' -',e)
        raise SystemExit(2)
    print('PHASE 1 SOURCE AUDIT: PASS')
    for p in REQUIRED: print(' OK',p.relative_to(ROOT))

if __name__=='__main__': main()
