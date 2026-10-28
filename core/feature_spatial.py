"""V3 60D spatial descriptors, deterministic for train and webcam.

Two hands x 30D, invariant to camera scale for body/hand distances.
All MediaPipe input is unmirrored image-space XYZ; only within-Hand uses z.
No distances are computed between Pose/Hand Z coordinate systems.
Missing hands/pose become masked zeros, not interpolated fake landmarks.
"""
from __future__ import annotations
import numpy as np

SPATIAL_DIM = 60
SPATIAL_VERSION = 'spatial60_v1'
# 30D per hand (see docs):
# 0:2 wrist XY/body; 2:4 palm center XY/body; 4:8 wrist distance to nose,
# same-side shoulder, chest, mouth; 8:13 5 curl;
# 13:16 palm normal XYZ; 16:18 palm axis XY;
# 18:23 tip to wrist scaled; 23:28 tip to palm scaled;
# 28:30 body-relative wrist velocity XY (per frame, no time normalization).
FINGERS=((2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20))
TIPS=(4,8,12,16,20)
EPS=1e-6


def _unit(x):
    d=float(np.linalg.norm(x))
    return np.asarray(x,dtype=np.float32)/(d+EPS)


def _bend(a,b,c):
    u=_unit(b-a); v=_unit(c-b)
    return float(np.arccos(np.clip(np.dot(u,v),-1.0,1.0))/np.pi)


def spatial60(left,right,pose,left_present,right_present,pose_present,
              timestamps=None,return_mask=False):
    l=np.asarray(left,np.float32);r=np.asarray(right,np.float32);p=np.asarray(pose,np.float32)
    if l.ndim!=3 or l.shape[1:]!=(21,3) or r.shape!=l.shape or p.shape!=(len(l),33,3):
        raise ValueError(f'Expected left,right [T,21,3], pose [T,33,3], got {l.shape},{r.shape},{p.shape}')
    T=len(l)
    if T==0:raise ValueError('Empty sequence')
    lp=np.asarray(left_present,bool).reshape(-1);rp=np.asarray(right_present,bool).reshape(-1)
    pp=np.asarray(pose_present,bool).reshape(-1)
    if any(len(z)!=T for z in (lp,rp,pp)):raise ValueError('Presence length mismatch')
    if timestamps is not None:
        ts=np.asarray(timestamps,dtype=np.float64).reshape(-1)
        if len(ts)!=T or not np.isfinite(ts).all() or np.any(np.diff(ts)<0):
            raise ValueError('timestamps invalid')
    if not (np.isfinite(l).all() and np.isfinite(r).all() and np.isfinite(p).all()):
        raise ValueError('Nonfinite landmark')
    out=np.zeros((T,60),np.float32)
    mask=np.zeros((T,60),bool)
    for t in range(T):
        if pp[t]:
            shoulders=p[t,[11,12],:2]
            center=shoulders.mean(axis=0)
            scale=float(np.linalg.norm(shoulders[0]-shoulders[1]))
            valid_pose=np.isfinite(scale) and scale>1e-4
            nose=p[t,0,:2]; mouth=p[t,[9,10],:2].mean(axis=0)
        else:
            valid_pose=False
        for hand_i,(xyz,present) in enumerate(((l,lp),(r,rp))):
            if not present[t]:continue
            hand=xyz[t];o=hand_i*30
            palm=hand[[0,5,9,13,17],:].mean(axis=0)
            hs=float(np.linalg.norm(hand[9]-hand[0]))
            if not np.isfinite(hs) or hs<=1e-5:continue
            # Body dependent geometry, XY only.
            if valid_pose:
                wrist=hand[0,:2]
                out[t,o:o+2]=(wrist-center)/scale
                out[t,o+2:o+4]=(palm[:2]-center)/scale
                anchor=[nose,shoulders[hand_i],center,mouth]
                out[t,o+4:o+8]=[np.linalg.norm(wrist-x)/scale for x in anchor]
                mask[t,o:o+8]=True
            # Hand-only shape: use hand coordinates only.
            curls=[]
            for ids in FINGERS:
                if len(ids)==3:
                    curls.append(_bend(hand[ids[0]],hand[ids[1]],hand[ids[2]]))
                else:
                    curls.append(.5*(_bend(hand[ids[0]],hand[ids[1]],hand[ids[2]])+
                                       _bend(hand[ids[1]],hand[ids[2]],hand[ids[3]])))
            out[t,o+8:o+13]=curls
            axis1=hand[5]-hand[0];axis2=hand[17]-hand[0]
            norm=np.cross(axis1,axis2)
            out[t,o+13:o+16]=_unit(norm)
            out[t,o+16:o+18]=_unit((hand[9]-hand[0])[:2])
            out[t,o+18:o+23]=[np.linalg.norm(hand[q]-hand[0])/hs for q in TIPS]
            out[t,o+23:o+28]=[np.linalg.norm(hand[q]-palm)/hs for q in TIPS]
            mask[t,o+8:o+28]=True
            # Velocity normalized by shoulder scale, consecutive valid frames only.
            if valid_pose and t>0 and present[t-1] and pp[t-1]:
                prev_shoulders=p[t-1,[11,12],:2]
                prev_scale=float(np.linalg.norm(prev_shoulders[0]-prev_shoulders[1]))
                if prev_scale>1e-4:
                    prev=(xyz[t-1,0,:2]-prev_shoulders.mean(axis=0))/prev_scale
                    curr=(hand[0,:2]-center)/scale
                    out[t,o+28:o+30]=curr-prev
                    mask[t,o+28:o+30]=True
    if not np.isfinite(out).all():raise ValueError('Spatial contains NaN/Inf')
    return (out,mask) if return_mask else out


def _pick(d,names,shape=None,optional=False):
    for n in names:
        if n in d.files:
            x=np.asarray(d[n]);break
    else:
        if optional:return None
        raise KeyError(f'Cannot find any {names} in landmarks.npz')
    if shape is not None and x.shape!=shape:
        raise ValueError(f'Unexpected landmark array shape for {names}: {x.shape}; expected {shape}')
    return x


def load_raw_npz(path):
    """Load Phase05/06 raw landmarks; missing pose is explicitly masked."""
    with np.load(path,allow_pickle=False) as d:
        l=_pick(d,('left_hand','left','left_hand_landmarks'))
        r=_pick(d,('right_hand','right','right_hand_landmarks'))
        if l.ndim!=3 or l.shape[1:]!=(21,3) or r.shape!=l.shape:
            raise ValueError(f'Bad hands shape {l.shape}/{r.shape}')
        T=len(l)
        p=_pick(d,('pose','pose_landmarks','pose_xyz'),optional=True)
        if p is None:p=np.zeros((T,33,3),np.float32)
        lp=_pick(d,('left_present','left_hand_present'),optional=True)
        rp=_pick(d,('right_present','right_hand_present'),optional=True)
        pp=_pick(d,('pose_present',),optional=True)
        if lp is None:lp=np.any(np.abs(l)>1e-8,axis=(1,2))
        if rp is None:rp=np.any(np.abs(r)>1e-8,axis=(1,2))
        if pp is None:pp=np.any(np.abs(p)>1e-8,axis=(1,2))
        ts=_pick(d,('timestamps',),optional=True)
    return (l,r,p,lp,rp,pp,ts)


def find_landmark_path(row,root):
    from pathlib import Path
    root=Path(root)
    candidates=[]
    if row.get('sample_path'):
        d=Path(row['sample_path']);d=d if d.is_absolute() else root/d
        candidates.extend([d/'landmarks.npz',d])
    d=root/'dataset/raw'/str(row.get('sign',''))/str(row.get('subject_id',''))/str(row.get('session_id',''))/str(row.get('sample_id',''))
    candidates.append(d/'landmarks.npz')
    for x in candidates:
        if x.is_file() and x.suffix=='.npz':return x
    raise FileNotFoundError(f'Raw landmarks.npz missing for {row.get("sample_id")}; candidates={candidates}')


def spatial_from_row(row,root):
    l,r,p,lp,rp,pp,ts=load_raw_npz(find_landmark_path(row,root))
    return spatial60(l,r,p,lp,rp,pp,ts)
