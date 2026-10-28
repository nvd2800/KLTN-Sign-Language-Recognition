"""Safely migrate diagnostic recordings to label/subject/session/condition/id.

Supports both historical layouts:
  label/subject/condition/timestamp
  label/subject/session/label_subject_session_NNN

New canonical layout:
  label/subject/session/condition/condition_subject_session_NNN

Default dry-run. --execute explicitly moves directories without altering the
underlying video/landmark/feature bytes. A separate user backup is recommended.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.feature_diagnostic_video import (
    check_trial_layout, next_sample_path, write_metadata, rebuild_trial_manifest,
)


def _checksum(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def plan(root: Path, default_session: str | None = None):
    root = Path(root)
    counters: dict[tuple[str, str, str, str], int] = {}
    actions, skipped = [], []
    files = sorted(set(root.glob('*/*/*/*/metadata.json')) |
                   set(root.glob('*/*/*/*/*/metadata.json')))
    for source in files:
        old = source.parent
        try:
            meta = json.loads(source.read_text(encoding='utf-8'))
            label, subject, session, condition = check_trial_layout(old, meta, root)
            # New canonical structure: already done, never move again.
            if len(old.relative_to(root).parts) == 5:
                continue
            session = session or default_session
            if not session:
                skipped.append({'path': str(old), 'reason': 'NO_SESSION; specify --default-session only when verified'})
                continue
            if not (old / 'raw_landmarks.npz').is_file() or not (old / 'features.npz').is_file():
                skipped.append({'path': str(old), 'reason': 'MISSING_NPZ'})
                continue
            new_parent = root / label / subject / session / condition
            key = (label, subject, session, condition)
            if key not in counters:
                _, initial = next_sample_path(root, label, subject, session, condition)
                counters[key] = int(initial.rsplit('_', 1)[-1]) - 1
            while True:
                counters[key] += 1
                name = f'{condition}_{subject}_{session}_{counters[key]:03d}'
                dest = new_parent / name
                if not dest.exists():
                    break
            actions.append({
                'src': str(old), 'dest': str(dest), 'condition': condition,
                'session': session, 'old_id': meta.get('sample_id', old.name),
                'new_id': name,
            })
        except (OSError, ValueError, TypeError) as exc:
            skipped.append({'path': str(old), 'reason': f'{type(exc).__name__}: {exc}'})
    if len({row['dest'] for row in actions}) != len(actions):
        raise RuntimeError('Duplicate destinations in migration plan')
    return actions, skipped


def migrate(root: Path, actions: list[dict]):
    """All-or-rollback directory move with original metadata/manifest restored."""
    root = Path(root)
    for row in actions:
        src, dest = Path(row['src']), Path(row['dest'])
        if not src.is_dir() or dest.exists():
            raise RuntimeError(f'Migration preflight failed: {src} => {dest}')
        if not (src/'metadata.json').is_file():
            raise RuntimeError(f'Missing metadata for {src}')
        if not (src/'raw_landmarks.npz').is_file() or not (src/'features.npz').is_file():
            raise RuntimeError(f'Missing NPZ for {src}')
    manifest_path = root/'manifest.csv'
    manifest_content = manifest_path.read_bytes() if manifest_path.is_file() else None
    moved = []
    try:
        for row in actions:
            src, dest = Path(row['src']), Path(row['dest'])
            original_meta = (src/'metadata.json').read_bytes()
            files = [p for p in src.iterdir() if p.is_file() and p.name != 'metadata.json']
            hashes = {p.name: _checksum(p) for p in files}
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved.append((src, dest, original_meta))
            meta = json.loads(original_meta.decode('utf-8'))
            meta.update({
                'legacy_sample_id': row['old_id'],
                'legacy_trial_dir': row['src'],
                'sample_id': row['new_id'],
                'session': row['session'],
                'session_id': row['session'],
                'condition': row['condition'],
            })
            write_metadata(dest, meta)
            for name, digest in hashes.items():
                if _checksum(dest/name) != digest:
                    raise RuntimeError(f'Integrity failure for migrated {dest/name}')
        result = rebuild_trial_manifest(root)
        if result['skipped']:
            # Skips may refer to unrelated incomplete historical samples;
            # retain migration success, but surface these for review.
            result['note'] = 'Some historical samples skipped; inspect skipped list.'
        return {'moved': [str(dst) for _, dst, _ in moved], 'manifest_result': result}
    except Exception:
        for src, dest, original_meta in reversed(moved):
            if dest.is_dir():
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), str(src))
                (src/'metadata.json').write_bytes(original_meta)
        if manifest_content is None:
            manifest_path.unlink(missing_ok=True)
        else:
            manifest_path.write_bytes(manifest_content)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='Migrate to label/subject/session/condition/sample')
    parser.add_argument('--input', default='logs/feature_diagnostic')
    parser.add_argument('--default-session', default=None,
                        help='Only for legacy samples whose real session is known; do not guess')
    parser.add_argument('--execute', action='store_true',
                        help='Explicit migration; make a backup before executing')
    args = parser.parse_args(argv)
    root = Path(args.input)
    if not root.is_absolute():
        root = ROOT / root
    if not root.is_dir():
        raise FileNotFoundError(root)
    actions, skipped = plan(root, args.default_session)
    print(json.dumps({'mode': 'EXECUTE' if args.execute else 'DRY_RUN',
                      'planned': len(actions), 'skipped': skipped,
                      'preview': actions[:20]}, ensure_ascii=False, indent=2))
    if args.execute:
        result = migrate(root, actions)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print('DRY RUN: no files changed. Back up logs/feature_diagnostic before --execute.')
    return actions, skipped

if __name__ == '__main__':
    main()
