"""Audit, preview and play original camera videos paired with signer/trial metadata.

Examples:
 python scripts/review_feature_diagnostic_videos.py --label xin_chao --subject S01 --contact-sheets
 python scripts/review_feature_diagnostic_videos.py --label xin_chao --subject S01 --play
 python scripts/review_feature_diagnostic_videos.py --all --contact-sheets

Historical trials without video remain present as LEGACY_VIDEO_MISSING.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.feature_diagnostic_video import probe_video, _sha256, VideoSaveError, check_trial_layout

REPORT_COLS = [
    'sample_id', 'label', 'subject', 'session',
    'condition', 'captured_at', 'frames_raw', 'video_frames',
    'video_fps', 'video_size_mb', 'any_hand_ratio', 'status',
    'problem', 'video_path', 'raw_path', 'features_path', 'metadata_path',
    'preview_path',
]


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def collect_rows(source_root: Path, label=None, subject=None, condition=None, session=None, deep_video_check=True):
    source_root = Path(source_root)
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    records = []
    metas = sorted(set(source_root.glob('*/*/*/*/metadata.json')) |
                   set(source_root.glob('*/*/*/*/*/metadata.json')))
    for path in metas:
        trial = path.parent
        row = dict.fromkeys(REPORT_COLS, '')
        problems = []
        try:
            meta = json.loads(path.read_text(encoding='utf-8'))
            sign, signer, sess, cond = check_trial_layout(trial, meta, source_root)
            if (label and sign != label) or (subject and signer != subject) or \
               (condition and cond != condition) or (session and sess != session):
                continue
            sid = trial.name
            row.update(sample_id=sid, label=sign, subject=signer,
                       condition=cond, metadata_path=str(path),
                       raw_path=str(trial/'raw_landmarks.npz'),
                       features_path=str(trial/'features.npz'))
            row.update(session=meta.get('session', meta.get('session_id', '')),
                       captured_at=meta.get('captured_at', ''),
                       frames_raw=meta.get('frames', ''),
                       any_hand_ratio=(meta.get('qc') or {}).get('any_hand_ratio', ''))
            if not (trial/'raw_landmarks.npz').is_file():
                problems.append('RAW_MISSING')
            if not (trial/'features.npz').is_file():
                problems.append('FEATURES_MISSING')
            video = meta.get('video', {}) or {}
            fname = video.get('filename')
            legacy = not video
            if not fname:
                possible = [trial/f for f in ('source_video.avi', 'source_video.mp4') if (trial/f).is_file()]
                if possible:
                    fname = possible[0].name
                elif video.get('status') == 'FAILED':
                    problems.append('VIDEO_ENCODING_FAILED')
                elif video.get('status') == 'DISABLED':
                    problems.append('VIDEO_DISABLED')
                else:
                    problems.append('LEGACY_VIDEO_MISSING' if legacy else 'VIDEO_MISSING')
            if fname:
                video_path = trial/fname
                row['video_path'] = str(video_path)
                if not video_path.is_file():
                    problems.append('VIDEO_FILE_MISSING')
                else:
                    if deep_video_check:
                        info = probe_video(video_path, expected_frames=meta.get('frames'), strict=True)
                        row['video_frames'] = info['decoded_frames']
                    else:
                        row['video_frames'] = video.get('decoded_frames', '')
                    row['video_fps'] = video.get('fps', '')
                    row['video_size_mb'] = round(video_path.stat().st_size / (1024*1024), 3)
                    if video.get('sha256') and _sha256(video_path) != video['sha256']:
                        problems.append('VIDEO_SHA256_MISMATCH')
        except (ValueError, TypeError, OSError, VideoSaveError, KeyError) as exc:
            problems.append(f'REVIEW_ERROR:{type(exc).__name__}:{str(exc)[:150]}')
        row['status'] = 'PASS' if not problems else 'REVIEW'
        row['problem'] = '|'.join(problems)
        records.append(row)
    return records


def render_contact_sheet(row, out_path: Path, samples=6):
    video = Path(row['video_path']) if row.get('video_path') else None
    if video is None or not video.is_file():
        return False
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return False
    shots = []
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = np.linspace(0, count-1, min(samples, count), dtype=int) if count else np.array([], dtype=int)
        for fidx in target:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, image = cap.read()
            if ok:
                shots.append((int(fidx), image))
        if not shots:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in range(samples):
                ok, image = cap.read()
                if not ok:
                    break
                shots.append((i, image))
    finally:
        cap.release()
    if not shots:
        return False
    tile_w, tile_h = 360, 225
    cols, rows = 3, (len(shots)+2)//3
    header_h = 67
    sheet = np.full((header_h + rows*tile_h, cols*tile_w, 3), (25, 31, 42), np.uint8)
    title = f"SIGN: {row['label']} | SIGNER: {row['subject']} | {row['condition']}"
    cv2.putText(sheet, title[:100], (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .55, (230,238,246), 1, cv2.LINE_AA)
    cv2.putText(sheet, f"SAMPLE: {row['sample_id']} | SESSION: {row['session']}",
                (12, 54), cv2.FONT_HERSHEY_SIMPLEX, .51, (190,210,239), 1, cv2.LINE_AA)
    for i, (fidx, frame) in enumerate(shots):
        h,w=frame.shape[:2]
        resized = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA if w>tile_w else cv2.INTER_LINEAR)
        y=header_h + (i//cols)*tile_h; x=(i%cols)*tile_w
        sheet[y:y+tile_h,x:x+tile_w]=resized
        cv2.rectangle(sheet,(x+3,y+3),(x+102,y+26),(20,26,36),-1)
        cv2.putText(sheet, f'Frame {fidx+1}', (x+8,y+21), cv2.FONT_HERSHEY_SIMPLEX,.48,(247,247,247),1)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    return bool(cv2.imwrite(str(out_path),sheet))


REVIEW_COLUMNS = ['sample_id','label','subject','session','condition',
                  'reviewer','decision','reviewed_at','video_path','metadata_path']


def record_manual_decision(review_csv: Path, row: dict, decision: str, reviewer: str):
    """Human annotation in a separate file: NEVER edits ground truth or raw source."""
    decision = decision.upper()
    if decision not in {'ACCEPT','REVIEW','REJECT'}:
        raise ValueError('decision must be ACCEPT, REVIEW or REJECT')
    if not reviewer.strip():
        raise ValueError('reviewer is required')
    review_csv = Path(review_csv)
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if review_csv.is_file():
        with review_csv.open('r',encoding='utf-8-sig',newline='') as f:
            existing = list(csv.DictReader(f))
    key = (row['sample_id'],row['label'],row['subject'])
    existing = [r for r in existing if (r['sample_id'],r['label'],r['subject']) != key]
    existing.append({field:value for field,value in {
        'sample_id':row['sample_id'],'label':row['label'],
        'subject':row['subject'],'session':row['session'],
        'condition':row['condition'],'reviewer':reviewer,
        'decision':decision,'reviewed_at':datetime.now(timezone.utc).isoformat(),
        'video_path':row['video_path'],'metadata_path':row['metadata_path'],
    }.items()})
    temp = review_csv.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=REVIEW_COLUMNS)
        writer.writeheader();writer.writerows(existing)
    temp.replace(review_csv)
    return review_csv


def play_trials(rows, review_csv: Path, reviewer: str):
    """Original footage in an OpenCV window; labels are preview-only."""
    for index,row in enumerate(rows,1):
        path = Path(row['video_path']) if row.get('video_path') else None
        if path is None or not path.is_file():
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print('VIDEO ERROR:',path)
            continue
        print(f'[{index}/{len(rows)}] {row["label"]} {row["subject"]} {row["condition"]} {row["sample_id"]}')
        paused = False
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        wait = max(1,round(1000/max(fps,1)))
        current = None
        try:
            while True:
                if not paused or current is None:
                    ok,frame=cap.read()
                    if not ok:
                        break
                    current=frame
                view=current.copy()
                cv2.rectangle(view,(0,0),(view.shape[1],65),(25,29,39),-1)
                s1=f"{row['label']} | {row['subject']} | {row['condition']}"
                cv2.putText(view,s1[:88],(12,26),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2)
                cv2.putText(view,f"{row['sample_id']} | A accept R review X reject N next Q quit",(12,53),
                            cv2.FONT_HERSHEY_SIMPLEX,.48,(210,222,235),1)
                cv2.imshow('KLTN Feature Diagnostic | Video Review',view)
                key=cv2.waitKey(0 if paused else wait)&255
                if key in (ord('q'),ord('Q'),27):
                    return
                if key in (ord('n'),ord('N')):
                    break
                if key in (ord('a'),ord('A'),ord('r'),ord('R'),ord('x'),ord('X')):
                    decision = {ord('a'):'ACCEPT',ord('r'):'REVIEW',ord('x'):'REJECT'}[ord(chr(key).lower())]
                    dest = record_manual_decision(review_csv,row,decision,reviewer)
                    print(f'[HUMAN {decision}] {row["sample_id"]} -> {dest}')
                    break
                if key==ord(' '):
                    paused=not paused
        finally:
            cap.release()
    cv2.destroyAllWindows()


def main(argv=None):
    ap=argparse.ArgumentParser(description='Audit source_video linked to signer and captured features')
    ap.add_argument('--input', default='logs/feature_diagnostic')
    ap.add_argument('--label',default=None)
    ap.add_argument('--subject',default=None)
    ap.add_argument('--condition',default=None)
    ap.add_argument('--session',default=None)
    ap.add_argument('--all',action='store_true',help='Explicitly review every label/signer')
    ap.add_argument('--contact-sheets', action='store_true',help='Generate 6-frame previews (JPG)')
    ap.add_argument('--play',action='store_true',help='OpenCV playback; A accept, R review, X reject, N next, Q quit')
    ap.add_argument('--reviewer',default='reviewer01',help='Who manually reviewed footage')
    ap.add_argument('--review-decisions',default='logs/feature_diagnostic/reports/manual_video_reviews.csv')
    ap.add_argument('--fast',action='store_true',help='Skip decoding all frames during audit')
    ap.add_argument('--output',default='logs/feature_diagnostic/reports')
    a=ap.parse_args(argv)
    if not a.all and not (a.label or a.subject):
        ap.error('Choose --label / --subject or --all to avoid accidentally auditing every signer')
    root=resolve(a.input)
    rows=collect_rows(root, label=a.label,subject=a.subject,condition=a.condition,session=a.session,
                      deep_video_check=not a.fast)
    out=resolve(a.output)
    out.mkdir(parents=True,exist_ok=True)
    suffix='_'.join(v for v in (a.label,a.subject,a.condition) if v) or 'ALL'
    if a.contact_sheets:
        for r in rows:
            video = Path(r['video_path']) if r['video_path'] else None
            if video is None or not video.is_file():
                continue
            preview=out/'previews'/r['label']/r['subject']/r['condition']/(r['sample_id']+'.jpg')
            if render_contact_sheet(r,preview):
                r['preview_path']=str(preview)
    csv_path=out/f'videos_{suffix}.csv'
    with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=REPORT_COLS)
        writer.writeheader();writer.writerows(rows)
    summary={'total':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),
             'problems':dict(Counter(err for r in rows for err in r['problem'].split('|') if err)),
             'by_signer':dict(Counter(r['subject'] for r in rows)),
             'report_csv':str(csv_path),'notes':[
                 'Existing legacy trials without footage are reported, never reconstructed.',
                 'Signer code is capture metadata, not proof of physical identity.',
                 'Do not publish unconsented raw videos, especially when faces are visible.',
             ]}
    json_path=out/f'videos_{suffix}.json'
    json_path.write_text(json.dumps({'summary':summary,'records':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    for r in rows:
        print(f"[{r['status']}] {r['label']}/{r['subject']}/{r['condition']}/{r['sample_id']} | video={Path(r['video_path']).name if r['video_path'] else 'MISSING'} | {r['problem']}")
    if a.play:
        try:
            play_trials(rows,resolve(a.review_decisions),a.reviewer)
        finally:
            cv2.destroyAllWindows()
    return summary


if __name__=='__main__':
    main()
