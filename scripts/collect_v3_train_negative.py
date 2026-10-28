"""Signer-safe webcam collector entry point. Only TRAIN signers allowed.

Wraps the already installed live_feature_diagnostic.py; the output is a
new, real-camera trial with video, never a synthetic copy of TEST samples.
"""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def validate_capture(label,subject,mapping,labels):
    subject=str(subject).upper().strip();label=str(label).strip()
    if mapping.get(subject)!='train':
        raise ValueError(f'{subject} is not assigned to TRAIN; do NOT collect TEST/VAL data for training!')
    if label in ('UNKNOWN','REST') or label not in labels:
        raise ValueError(f'{label} must be a known sign class other than REST/UNKNOWN')
    return label,subject


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',required=True)
    p.add_argument('--subject',required=True)
    p.add_argument('--frames',type=int,default=32)
    p.add_argument('--known-summary',default='dataset/processed/temporal_phase7/dataset_summary.json')
    p.add_argument('--labels',default='dataset/processed/temporal_v3_spatial/labels.json')
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--session',default='SES01',help='Recording session id')
    p.add_argument('--video-format',choices=('avi','mp4'),default='avi')
    p.add_argument('--fullscreen',action='store_true')
    a=p.parse_args()
    def rp(s):
        q=Path(s);return q if q.is_absolute() else ROOT/q
    from scripts.prepare_v3_spatial import signer_mapping
    mapping=signer_mapping(rp(a.known_summary))
    labs=json.loads(rp(a.labels).read_text(encoding='utf-8'))['labels']
    label,subject=validate_capture(a.label,a.subject,mapping,labs)
    if a.frames<8:raise ValueError('At least 8 frames required')
    collector=ROOT/'scripts/live_feature_diagnostic.py'
    if not collector.is_file():raise FileNotFoundError(f'Install previous Diagnostic collector: {collector}')
    cmd=[sys.executable,str(collector),'--label',label,'--subject',subject,
         '--frames',str(a.frames),'--camera',str(a.camera),'--save-video',
         '--session',a.session,'--video-format',a.video_format]
    if a.fullscreen:cmd.append('--fullscreen')
    print('Collecting REAL hard negatives from TRAIN signer ONLY:',subject,label)
    print('Press 2=wrong_position, 3=wrong_handshape, 4=wrong_orientation, 5=partial, Q=quit')
    print('Command:', ' '.join(cmd))
    subprocess.run(cmd,cwd=str(ROOT),check=True)

if __name__=='__main__':main()
