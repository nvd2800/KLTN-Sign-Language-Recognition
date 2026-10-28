"""Video/provenance helpers for Feature Diagnostic.

Each trial keeps unmirrored ORIGINAL camera pixels, exactly aligned in capture
order to ``raw_landmarks.npz``. No overlays or mirror transforms are burned into
training videos; annotations are rendered only by the separate reviewer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
import statistics
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

MANIFEST_COLUMNS = [
    'sample_id', 'captured_at', 'label', 'subject',
    'session', 'condition', 'frames_raw', 'video_status', 'video_file',
    'video_frames', 'video_fps', 'video_sha256', 'video_bytes',
    'any_hand_ratio', 'pose_ratio', 'trial_dir',
]


class VideoSaveError(RuntimeError):
    pass


DIAGNOSTIC_CONDITIONS = (
    'correct', 'wrong_position', 'wrong_handshape',
    'wrong_orientation', 'partial', 'correct_repeat',
)


def next_sample_path(root: Path, label: str, subject: str, session: str,
                     condition: str) -> tuple[Path, str]:
    """Allocate label/Sxx/SESxx/condition/condition_Sxx_SESxx_NNN.

    Counter is per (label, signer, session, condition), not shared between
    conditions. A new capture never overwrites an existing directory.
    """
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', v) for v in (label, subject, session)):
        raise ValueError('Invalid label / subject / session')
    if condition not in DIAGNOSTIC_CONDITIONS:
        raise ValueError(f'Unsupported diagnostic condition: {condition}')
    parent = Path(root) / label / subject / session / condition
    prefix = f'{condition}_{subject}_{session}_'
    last = 0
    for trial in parent.iterdir() if parent.is_dir() else ():
        if trial.is_dir() and trial.name.startswith(prefix):
            suffix = trial.name[len(prefix):]
            if suffix.isdigit():
                last = max(last, int(suffix))
    number = last + 1
    while (parent / f'{prefix}{number:03d}').exists():
        number += 1
    name = f'{prefix}{number:03d}'
    return parent / name, name


def check_trial_layout(trial: Path, metadata: dict, root: Path) -> tuple[str, str, str, str]:
    """Validate diagnostic sample paths and identity metadata.

    Read-only compatibility with older layouts:
      label/subject/condition/timestamp
      label/subject/session/label_subject_session_NNN
    NEW required layout:
      label/subject/session/condition/condition_subject_session_NNN
    """
    try:
        relative = Path(trial).resolve().relative_to(Path(root).resolve())
    except ValueError as error:
        raise ValueError('Trial is outside diagnostic root') from error
    parts = relative.parts
    if len(parts) not in (4, 5):
        raise ValueError(f'Expected 4 (legacy) or 5 (new) path components, got {relative}')
    label, subject = parts[:2]
    sample_id = parts[-1]
    cond = str(metadata.get('condition') or '')
    session = str(metadata.get('session') or metadata.get('session_id') or '')
    if cond not in DIAGNOSTIC_CONDITIONS:
        raise ValueError(f'Invalid diagnostic condition in metadata: {cond}')
    for field, expected in (('label', label), ('subject', subject),
                            ('subject_id', subject), ('sample_id', sample_id)):
        if metadata.get(field) and str(metadata[field]) != expected:
            raise ValueError(f'Metadata/path mismatch for {field}: {metadata[field]} != {expected}')
    if len(parts) == 5:
        _, _, folder_session, folder_condition, _ = parts
        if not session or folder_session != session or folder_condition != cond:
            raise ValueError('Session/condition folder does not match metadata')
        prefix = f'{cond}_{subject}_{session}_'
        if not sample_id.startswith(prefix) or not sample_id[len(prefix):].isdigit():
            raise ValueError(f'Sample folder must match {prefix}NNN, got {sample_id}')
        return label, subject, session, cond
    third = parts[2]
    if third == cond:
        return label, subject, session, cond  # timestamp under legacy condition
    if not session or third != session:
        raise ValueError(f'Legacy session folder {third} does not match metadata session {session}')
    prefix = f'{label}_{subject}_{session}_'
    if not sample_id.startswith(prefix) or not sample_id[len(prefix):].isdigit():
        raise ValueError(f'Legacy sample must match {prefix}NNN, got {sample_id}')
    return label, subject, session, cond


def stable_capture_fps(timestamps, fallback=30.0):
    """Playback FPS based on MediaPipe input-frame timestamps, not model FPS."""
    a = np.asarray(timestamps, dtype=np.float64)
    if a.ndim != 1 or len(a) < 2 or not np.all(np.isfinite(a)):
        return float(fallback)
    deltas = np.diff(a)
    if np.any(deltas <= 0):
        raise ValueError('Video timestamps must be strictly increasing')
    duration = float(a[-1] - a[0])
    fps = (len(a) - 1) / duration if duration > 0 else fallback
    return float(min(60.0, max(1.0, fps)))


def _sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path: Path, expected_frames=None, strict=True):
    """Actually decode frames; a file existing is NOT enough to verify a video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoSaveError(f'Video cannot be opened: {path}')
    decoded = 0
    width = height = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise VideoSaveError(f'Invalid decoded frame in {path}')
            height, width = frame.shape[:2]
            decoded += 1
    finally:
        cap.release()
    if decoded == 0:
        raise VideoSaveError(f'Zero decodable video frames: {path}')
    if expected_frames is not None and strict and decoded != int(expected_frames):
        raise VideoSaveError(f'Video-frame mismatch: decoded={decoded}, expected={expected_frames}')
    return {'decoded_frames': decoded, 'width': width, 'height': height}


def save_source_video(trial_dir: Path, frames, timestamps, preferred='avi'):
    """Write verified, unmirrored, unannotated frames alongside saved landmarks.

    Default AVI/MJPG is deliberately preferred for reliable local OpenCV
    playback on Windows. Browser playback is not assumed. MP4 can be requested.
    On failed encoders, fallback to AVI/MJPG then AVI/XVID.
    """
    if preferred not in ('avi', 'mp4'):
        raise ValueError('preferred must be avi or mp4')
    if not frames:
        raise VideoSaveError('No frames to write')
    h, w = frames[0].shape[:2]
    for i, frame in enumerate(frames):
        if frame.shape != (h, w, 3) or frame.dtype != np.uint8:
            raise VideoSaveError(f'Unexpected camera frame at {i}: shape={frame.shape}, dtype={frame.dtype}')
    fps = stable_capture_fps(timestamps)
    candidates = ([('source_video.mp4', 'mp4v')] if preferred == 'mp4' else []) + [
        ('source_video.avi', 'MJPG'),
        ('source_video.avi', 'XVID'),
    ]
    trial_dir = Path(trial_dir)
    errors = []
    for file_name, codec in candidates:
        dest = trial_dir / file_name
        dest.unlink(missing_ok=True)
        writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        try:
            if not writer.isOpened():
                errors.append(f'{codec}: VideoWriter not opened')
                continue
            for frame in frames:
                writer.write(frame)
        except Exception as e:
            errors.append(f'{codec}: {type(e).__name__}: {e}')
            continue
        finally:
            writer.release()
        try:
            actual = probe_video(dest, expected_frames=len(frames))
            size = dest.stat().st_size
            if size < 128:
                raise VideoSaveError('Empty/too-small file')
            return {
                'status': 'SAVED_VERIFIED',
                'filename': file_name,
                'codec': codec,
                'fps': fps,
                'frame_count': len(frames),
                'decoded_frames': actual['decoded_frames'],
                'width': w,
                'height': h,
                'duration_seconds': float((len(timestamps) - 1) / fps),
                'bytes': size,
                'sha256': _sha256(dest),
                'pixels': 'ORIGINAL_UNMIRRORED_NO_OVERLAY',
                'saved_at': datetime.now(timezone.utc).isoformat(),
            }
        except (OSError, VideoSaveError) as e:
            errors.append(f'{codec}: verify failed: {e}')
        dest.unlink(missing_ok=True)
    raise VideoSaveError('All video codecs failed; ' + '; '.join(errors))


def write_metadata(trial_dir: Path, metadata: dict):
    """Replace only the existing trial's metadata atomically, never raw/features."""
    dest = Path(trial_dir) / 'metadata.json'
    temp = dest.with_suffix('.json.tmp')
    temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, dest)


def _manifest_row(trial_dir: Path, meta: dict):
    """One row per trial: support both old condition and new session directory."""
    trial_dir = Path(trial_dir)
    video = meta.get('video') or {}
    qc = meta.get('qc') or {}
    video_file = video.get('filename')
    video_path = trial_dir / video_file if video_file else None
    if not video_file:
        for name in ('source_video.avi', 'source_video.mp4'):
            if (trial_dir / name).is_file():
                video_file = name
                video_path = trial_dir / name
                break
    # Only metadata defines condition in the new layout, not parent name.
    label = str(meta.get('label') or trial_dir.parents[3 if trial_dir.parent.name in DIAGNOSTIC_CONDITIONS else 2].name)
    subject = str(meta.get('subject') or meta.get('subject_id') or trial_dir.parents[2 if trial_dir.parent.name in DIAGNOSTIC_CONDITIONS else 1].name)
    session = str(meta.get('session') or meta.get('session_id') or '')
    condition = str(meta.get('condition') or '')
    status = video.get('status') or ('LEGACY_VIDEO_MISSING' if not video_file else 'LEGACY_VIDEO_PRESENT')
    return {
        'sample_id': meta.get('sample_id') or trial_dir.name,
        'captured_at': meta.get('captured_at', ''),
        'label': label,
        'subject': subject,
        'session': session,
        'condition': condition,
        'frames_raw': meta.get('frames', ''),
        'video_status': status,
        'video_file': str(video_path) if video_path is not None else '',
        'video_frames': video.get('decoded_frames', ''),
        'video_fps': video.get('fps', ''),
        'video_sha256': video.get('sha256', ''),
        'video_bytes': video.get('bytes', ''),
        'any_hand_ratio': qc.get('any_hand_ratio', ''),
        'pose_ratio': qc.get('pose_ratio', ''),
        'trial_dir': str(trial_dir),
    }


def _read_existing_manifest(path: Path):
    if not path.is_file() or path.stat().st_size==0:
        return [], False
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        cols=reader.fieldnames or []
        if cols == MANIFEST_COLUMNS:
            return list(reader), False
        # Auto-migrate just the prior V2 manifest: drop alias while retaining
        # every other field; a backup is created before the write.
        if [c for c in cols if c != 'signer_alias'] == MANIFEST_COLUMNS and 'signer_alias' in cols:
            return [{k:r.get(k,'') for k in MANIFEST_COLUMNS} for r in reader], True
        raise ValueError(f'Unknown manifest columns in {path}. Back up/inspect before collecting; no data changed.')


def _write_manifest(path: Path, rows: list[dict], legacy_schema=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if legacy_schema and path.exists():
        # Preserve existing old-schema CSV byte-for-byte, including aliases,
        # without deleting any historical records.
        backup=path.with_name(path.stem+'.v2_alias_backup.csv')
        if backup.exists():
            backup=path.with_name(path.stem+'.v2_alias_backup_'+uuid.uuid4().hex[:8]+'.csv')
        shutil.copy2(path,backup)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex[:8]+'.tmp')
    try:
        with tmp.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=MANIFEST_COLUMNS,extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def append_trial_manifest(data_root: Path, trial_dir: Path, meta: dict):
    """Upsert a trial by its directory, preserving previous rows and sessions."""
    data_root=Path(data_root)
    path=data_root/'manifest.csv'
    rows,legacy_schema=_read_existing_manifest(path)
    newrow=_manifest_row(trial_dir,meta)
    normalized=str(Path(newrow['trial_dir']).resolve())
    rows=[r for r in rows if str(Path(r['trial_dir']).resolve())!=normalized]
    rows.append(newrow)
    _write_manifest(path,rows,legacy_schema=legacy_schema)
    return path


def rebuild_trial_manifest(data_root: Path):
    """Rebuild combined legacy+session manifest; do NOT alter existing files."""
    data_root = Path(data_root)
    path = data_root / 'manifest.csv'
    _, legacy_schema = _read_existing_manifest(path)
    # Rebuild from actual on-disk recordings only, so migrations remove stale
    # manifest paths. Preserve original NPZ/video and metadata without edits.
    keyed = {}
    scanned = 0
    skipped = []
    candidates = sorted(set(data_root.glob('*/*/*/*/metadata.json')) |
                        set(data_root.glob('*/*/*/*/*/metadata.json')))
    for source in candidates:
        trial = source.parent
        try:
            meta = json.loads(source.read_text(encoding='utf-8'))
            check_trial_layout(trial, meta, data_root)
            if not (trial / 'raw_landmarks.npz').is_file() or not (trial / 'features.npz').is_file():
                skipped.append(str(source) + ' [missing npz]')
                continue
            keyed[str(trial.resolve())] = _manifest_row(trial, meta)
            scanned += 1
        except (OSError, TypeError, ValueError) as error:
            skipped.append(str(source) + ' [' + str(error) + ']')
    _write_manifest(path, list(keyed.values()), legacy_schema=legacy_schema)
    return {'manifest': str(path), 'scanned': scanned,
            'total': len(keyed), 'skipped': skipped}
