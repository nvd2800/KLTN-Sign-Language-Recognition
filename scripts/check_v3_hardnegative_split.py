"""Read-only diagnostic for V3/V3.1 hard-negative preparation.

Run from project root:
    python scripts/check_v3_hardnegative_split.py

No PowerShell here-string required. Compatible with legacy and new Session/Condition layouts.
Does not move, rewrite, or train on any samples.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

NEGATIVE_CONDITIONS = ('wrong_position', 'wrong_handshape', 'wrong_orientation', 'partial')
KNOWN_POSITIVE_CONDITIONS = ('correct', 'correct_repeat')
EXPECTED_FILES = ('features.npz', 'raw_landmarks.npz')
FALLBACK_SPLITS = {
    'train': [f'S{i:02d}' for i in range(1, 8)],
    'val': ['S08'],
    'test': ['S09', 'S10'],
}


def load_mapping(path: Path):
    if not path.exists():
        return {s: split for split, subjects in FALLBACK_SPLITS.items() for s in subjects}, 'FALLBACK'
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('subjects'), dict):
        raise ValueError(f'Expected dataset_summary.json with subjects by split: {path}')
    mapping = {}
    for split, users in data['subjects'].items():
        if split not in ('train', 'val', 'test'):
            continue
        for subject in users:
            subject = str(subject).strip().upper()
            if subject in mapping and mapping[subject] != split:
                raise ValueError(f'Signer {subject} in BOTH {mapping[subject]} and {split}')
            mapping[subject] = split
    if not mapping:
        raise ValueError(f'No signer mapping in {path}')
    return mapping, 'DATASET_SUMMARY'


def parse_one(metadata_path: Path, root: Path, mapping: dict):
    rel = metadata_path.parent.relative_to(root)
    parts = rel.parts
    result = {'path': str(rel).replace('\\', '/'), 'status': 'SKIP', 'reason': '',
              'split': '', 'subject': '', 'condition': '', 'label': '', 'session': ''}
    try:
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        if not isinstance(meta, dict):
            raise ValueError('metadata must be JSON object')
    except (ValueError, OSError) as exc:
        result.update(status='INVALID', reason=f'metadata unreadable: {exc}')
        return result

    # For the new directory layout: label/Sxx/SESxx/condition/sample.
    # Match prepare_v3_spatial.py exactly: condition is read ONLY from metadata;
    # new layouts must explicitly record signer and label in metadata as well.
    result['label'] = str(meta.get('label') or meta.get('sign') or '').strip()
    subject = str(meta.get('subject') or meta.get('subject_id') or '').strip().upper()
    result['subject'] = subject
    result['session'] = str(meta.get('session') or meta.get('session_id') or (parts[2] if len(parts) == 5 else '')).strip()
    condition = str(meta.get('condition') or '').strip()
    result['condition'] = condition
    if condition in KNOWN_POSITIVE_CONDITIONS:
        result.update(status='SKIP', reason='correct/correct_repeat: positive, not hard negative')
        return result
    if condition not in NEGATIVE_CONDITIONS:
        result.update(status='INVALID', reason=f'condition missing/invalid: {condition!r}')
        return result
    if not result['label']:
        result.update(status='INVALID', reason='label/sign missing in metadata')
        return result
    if not subject:
        result.update(status='INVALID', reason='subject/subject_id missing in metadata')
        return result
    split = mapping.get(subject, '')
    result['split'] = split
    if not split:
        result.update(status='INVALID', reason=f'subject {subject!r} missing from signer mapping')
        return result
    missing = [fn for fn in EXPECTED_FILES if not (metadata_path.parent / fn).is_file()]
    if missing:
        result.update(status='INVALID', reason='missing ' + ', '.join(missing))
        return result
    # Do not trust path alone: ensure metadata is consistent with directory.
    if len(parts) == 5:
        label_dir, subject_dir, session_dir, condition_dir, sample_dir = parts
        if subject_dir.upper() != subject:
            result.update(status='INVALID', reason=f'subject mismatch path={subject_dir} metadata={subject}')
            return result
        if condition_dir != condition:
            result.update(status='INVALID', reason=f'condition mismatch path={condition_dir} metadata={condition}')
            return result
        if label_dir != result['label']:
            result.update(status='INVALID', reason=f'label mismatch path={label_dir} metadata={result["label"]}')
            return result
        prefix = f'{condition}_{subject}_{session_dir}_'
        if not (sample_dir.startswith(prefix) and sample_dir[len(prefix):].isdigit()):
            result.update(status='INVALID', reason='sample folder name does not match condition_Sxx_SESxx_NNN')
            return result
        if result['session'] != session_dir:
            result.update(status='INVALID', reason='session mismatch between path and metadata')
            return result
    elif len(parts) == 4:
        # Older layouts remain readable: label/Sxx/condition/timestamp OR
        # label/Sxx/SESxx/label_Sxx_SESxx_NNN.
        if parts[1].upper() != subject:
            result.update(status='INVALID', reason='subject mismatch in legacy path')
            return result
        if parts[2] not in (condition, result['session']):
            result.update(status='INVALID', reason='unrecognized legacy condition/session folder')
            return result
    else:
        result.update(status='INVALID', reason=f'path layout has {len(parts)} components; expected 4 or 5')
        return result
    result.update(status='ELIGIBLE', reason='has required files and metadata')
    return result


def scan(root: Path, mapping: dict):
    results = []
    for path in sorted(root.rglob('metadata.json')) if root.is_dir() else []:
        try:
            results.append(parse_one(path, root, mapping))
        except Exception as exc:
            results.append({'path': str(path), 'status': 'INVALID', 'reason': str(exc),
                            'split': '', 'subject': '', 'condition': '', 'label': '', 'session': ''})
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description='Check V3/V3.1 real hard negative splits (read-only)')
    ap.add_argument('--trials', default='logs/feature_diagnostic')
    ap.add_argument('--known-summary', default='dataset/processed/temporal_phase7/dataset_summary.json')
    ap.add_argument('--output', default='logs/v3_hardnegative_audit/split_check.json')
    a = ap.parse_args(argv)
    root = Path(a.trials)
    summary = Path(a.known_summary)
    mapping, mapping_source = load_mapping(summary)
    records = scan(root, mapping)
    eligible = Counter((row['split'], row['subject'], row['condition']) for row in records if row['status'] == 'ELIGIBLE')
    invalid = [row for row in records if row['status'] == 'INVALID']
    skipped = [row for row in records if row['status'] == 'SKIP']
    totals = {split: sum(n for (sp, _, _), n in eligible.items() if sp == split) for split in ('train','val','test')}
    print('\n=== V3 HARD NEGATIVE SPLIT AUDIT ===')
    print('Source:', root.resolve())
    print('Signer mapping:', mapping_source, '-', summary)
    print('Metadata files:', len(records), '| Positive skipped:', len(skipped), '| Invalid:', len(invalid))
    for split in ('train', 'val', 'test'):
        print(f'\n[{split.upper()}] total valid hard negatives = {totals[split]}')
        for subj in sorted(s for s, sp in mapping.items() if sp == split):
            count = sum(eligible[(split, subj, cond)] for cond in NEGATIVE_CONDITIONS)
            if count == 0:
                print(f'  {subj}: 0  <-- MISSING')
            else:
                details = ', '.join(f'{cond}={eligible[(split, subj, cond)]}' for cond in NEGATIVE_CONDITIONS if eligible[(split, subj, cond)])
                print(f'  {subj}: {count}  ({details})')
    if invalid:
        print('\n[INVALID] first 20 records:')
        for row in invalid[:20]:
            print(' ', row['path'], ':', row['reason'])
    verdict = ('STRICT_COUNTS_OK' if totals['train'] > 0 and totals['val'] > 0 and totals['test'] > 0
               else 'STRICT_BLOCKED_MISSING_SPLIT')
    print('\nVERDICT:', verdict)
    if verdict != 'STRICT_COUNTS_OK':
        print('Collect/repair real hard negatives for splits with 0; do not disable STRICT for final evaluation.')
    else:
        print('Split counts look sufficient. If prepare still fails, check NPZ shapes/keys and known dataset.')
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'verdict': verdict, 'source': str(root.resolve()), 'mapping_source': mapping_source,
              'totals': totals, 'eligible_by_split_subject_condition': [
                  {'split': sp, 'subject': su, 'condition': co, 'n': n}
                  for (sp, su, co), n in sorted(eligible.items())],
              'invalid': invalid, 'records': records}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Report:', output.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
