from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import numpy as np
import yaml

@dataclass(frozen=True)
class Phase4Config:
    wrist_index:int=0
    middle_mcp_index:int=9
    scale_epsilon:float=1e-6
    invalidate_near_zero_scale:bool=True
    max_abs_normalized_value:float=50.0

def load_phase4_config(path:Path):
    raw=yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    n=raw.get("normalization",{}) or {}
    cfg=Phase4Config(
        wrist_index=int(n.get("wrist_index",0)),
        middle_mcp_index=int(n.get("middle_mcp_index",9)),
        scale_epsilon=float(n.get("scale_epsilon",1e-6)),
        invalidate_near_zero_scale=bool(n.get("invalidate_near_zero_scale",True)),
        max_abs_normalized_value=float(n.get("max_abs_normalized_value",50.0)),
    )
    if not 0<=cfg.wrist_index<21 or not 0<=cfg.middle_mcp_index<21:
        raise ValueError("Landmark index must be in [0,20]")
    if cfg.wrist_index==cfg.middle_mcp_index:
        raise ValueError("wrist_index and middle_mcp_index must differ")
    if cfg.scale_epsilon<=0:
        raise ValueError("scale_epsilon must be > 0")
    return cfg,raw

def _check_hand(x,T,name):
    if x.shape!=(T,21,3):
        raise ValueError(f"{name} shape {x.shape}, expected {(T,21,3)}")

def normalize_hand_xyz(hand,present,cfg):
    hand=np.asarray(hand,dtype=np.float32)
    present=np.asarray(present,dtype=bool).reshape(-1)
    T=hand.shape[0]
    _check_hand(hand,T,"hand")
    if len(present)!=T:
        raise ValueError("presence mask length mismatch")
    out=np.zeros_like(hand,dtype=np.float32)
    scale=np.zeros(T,dtype=np.float32)
    finite=np.all(np.isfinite(hand),axis=(1,2))
    candidate=present & finite
    if T:
        wrist=hand[:,cfg.wrist_index,:]
        mcp=hand[:,cfg.middle_mcp_index,:]
        scale[:]=np.linalg.norm(mcp-wrist,axis=1).astype(np.float32)
    valid=candidate & np.isfinite(scale) & (scale>cfg.scale_epsilon)
    if np.any(valid):
        centered=hand[valid]-hand[valid,cfg.wrist_index,:][:,None,:]
        out[valid]=centered/scale[valid,None,None]
    output_present=valid if cfg.invalidate_near_zero_scale else candidate
    out[~output_present]=0.0
    return {
        "normalized":out,
        "input_present":present.astype(np.bool_),
        "scale":scale,
        "scale_valid":valid.astype(np.bool_),
        "output_present":output_present.astype(np.bool_),
    }

def build_phase4_features(left_hand,right_hand,left_present,right_present,cfg):
    left=np.asarray(left_hand,dtype=np.float32)
    right=np.asarray(right_hand,dtype=np.float32)
    T=left.shape[0]
    _check_hand(left,T,"left_hand"); _check_hand(right,T,"right_hand")
    L=normalize_hand_xyz(left,left_present,cfg)
    R=normalize_hand_xyz(right,right_present,cfg)
    xyz126=np.concatenate([L["normalized"].reshape(T,63),R["normalized"].reshape(T,63)],axis=1).astype(np.float32)
    presence=np.stack([L["output_present"],R["output_present"]],axis=1)
    return {
        "normalized_xyz126":xyz126,
        "left_normalized_xyz":L["normalized"],
        "right_normalized_xyz":R["normalized"],
        "hand_presence2":presence.astype(np.bool_),
        "left_present":L["output_present"],
        "right_present":R["output_present"],
        "left_input_present":L["input_present"],
        "right_input_present":R["input_present"],
        "scale2":np.stack([L["scale"],R["scale"]],axis=1).astype(np.float32),
        "scale_valid2":np.stack([L["scale_valid"],R["scale_valid"]],axis=1).astype(np.bool_),
    }

def inspect_feature_arrays(arrays,cfg):
    x=arrays["normalized_xyz126"]
    left=arrays["left_normalized_xyz"]; right=arrays["right_normalized_xyz"]
    p=arrays["hand_presence2"]; sv=arrays["scale_valid2"]
    T=x.shape[0]; issues=[]
    if x.shape!=(T,126): issues.append(f"bad_xyz126_shape:{x.shape}")
    if left.shape!=(T,21,3): issues.append(f"bad_left_shape:{left.shape}")
    if right.shape!=(T,21,3): issues.append(f"bad_right_shape:{right.shape}")
    if p.shape!=(T,2): issues.append(f"bad_presence_shape:{p.shape}")
    finite=bool(np.isfinite(x).all())
    if not finite: issues.append("non_finite_features")
    max_abs=float(np.max(np.abs(x))) if x.size else 0.0
    if max_abs>cfg.max_abs_normalized_value:
        issues.append(f"normalized_value_too_large:{max_abs:.6f}")
    wrist_err=0.0
    for hand,mask in [(left,p[:,0]),(right,p[:,1])]:
        if np.any(mask):
            wrist_err=max(wrist_err,float(np.max(np.abs(hand[mask,cfg.wrist_index,:]))))
    if wrist_err>1e-5: issues.append(f"wrist_not_zero:{wrist_err:.8f}")
    input_p=np.stack([arrays["left_input_present"],arrays["right_input_present"]],axis=1)
    invalid_scale=int(np.sum(input_p & ~sv))
    return {
        "frame_count":T,
        "feature_dim":126,
        "finite":finite,
        "max_abs_normalized":max_abs,
        "max_wrist_abs":wrist_err,
        "left_valid_ratio":float(p[:,0].mean()) if T else 0.0,
        "right_valid_ratio":float(p[:,1].mean()) if T else 0.0,
        "any_hand_valid_ratio":float(np.any(p,axis=1).mean()) if T else 0.0,
        "invalid_scale_hand_frames":invalid_scale,
        "issues":issues,
        "pass":not issues,
    }

def save_feature_sample(output_dir,arrays,metadata):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    npz=output_dir/"features.npz"; meta=output_dir/"feature_metadata.json"
    np.savez_compressed(npz,**arrays)
    meta.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    return npz,meta
