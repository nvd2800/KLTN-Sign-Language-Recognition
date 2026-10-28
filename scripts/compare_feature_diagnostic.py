"""Compare recorded paired diagnostic clips without webcam or TensorFlow."""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.spatial_handshape_diagnostic import pairwise_report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',default='xin_chao')
    p.add_argument('--subject',default='S01')
    p.add_argument('--input',default='logs/feature_diagnostic')
    p.add_argument('--output',default='logs/feature_diagnostic/reports')
    a=p.parse_args()
    rp=lambda s: Path(s) if Path(s).is_absolute() else ROOT/s
    rows,path=pairwise_report(rp(a.input),a.label,a.subject,rp(a.output))
    print('\nSPATIAL/HANDSHAPE FEATURE DIAGNOSTIC')
    print('Report CSV:',path)
    if not rows:print('Need at least one variant beyond the baseline');return
    important=['hand_position126','finger_angles30','distance9_legacy','pose27',
               'hand_body_position12','finger_curl10','palm_orientation6','wrist_relative_velocity4']
    for condition in sorted(set(r['condition'] for r in rows)):
        print('\n',condition.upper())
        for key in important:
            selected=[r for r in rows if r['condition']==condition and r['group']==key]
            if not selected:continue
            r=selected[-1]
            print(f"  {key:<29} RMS={r['rms'] if r['rms'] is not None else 'N/A'}  coverage={r['compared_ratio']:.2f}")
    print('\nNOTE: Higher RMS means more feature change, not automatic classification success.')

if __name__=='__main__':main()
