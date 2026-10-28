"""Optional visual review: one independent montage per TEST error's original video.

Does NOT do OCR or modify video; skips when source_video is unavailable.
Requires opencv-python installed on user's Windows machine.
"""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def make_sheet(video_path, out_path, count=6):
    import cv2,numpy as np
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():return False
    try:
        frames=[]
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n>0:
            indices=np.rint(np.linspace(0,n-1,count)).astype(int)
            for ix in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(ix))
                ok,img=cap.read()
                if ok:frames.append(img)
        if not frames:
            # Streaming fallback (no frame seek).
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            idx=0
            while len(frames)<count and idx<500:
                ok,img=cap.read()
                if not ok:break
                if idx%10==0:frames.append(img)
                idx+=1
        if not frames:return False
        thumb=[]
        for i,img in enumerate(frames):
            h,w=img.shape[:2];height=225;width=max(1,round(w*height/h))
            smaller=cv2.resize(img,(width,height),interpolation=cv2.INTER_AREA)
            cv2.putText(smaller,f'Frame {i+1}/{len(frames)}',(10,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
            thumb.append(smaller)
        cols=3;rows=(len(thumb)+cols-1)//cols
        tile_w=max(t.shape[1] for t in thumb)
        montage=np.zeros((rows*225,cols*tile_w,3),dtype=np.uint8)
        for i,t in enumerate(thumb):montage[i//cols*225:(i//cols+1)*225,i%cols*tile_w:i%cols*tile_w+t.shape[1]]=t
        out_path.parent.mkdir(parents=True,exist_ok=True)
        return bool(cv2.imwrite(str(out_path),montage))
    finally:cap.release()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--review',default='logs/v3_test_error_review/error_review.csv')
    p.add_argument('--out',default='logs/v3_test_error_review/contact_sheets')
    a=p.parse_args()
    def rp(x):
        x=Path(x);return x if x.is_absolute() else ROOT/x
    with rp(a.review).open('r',encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    ok=0
    for i,r in enumerate(rows):
        video=Path(r['video_path']) if r.get('video_path') else None
        if video is None or not video.is_file():
            print('SKIP no video:',r['sample_id']);continue
        # avoid using signer/sample_id as untrusted path
        out=rp(a.out)/f'error_{i+1:02d}.jpg'
        if make_sheet(video,out):ok+=1;print('Saved:',out)
        else:print('FAILED to read:',video)
    print('Rendered',ok,'/',len(rows),'error videos. No video was modified.')

if __name__=='__main__':main()
