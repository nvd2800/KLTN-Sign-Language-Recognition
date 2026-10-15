from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class Phase6Config:
    # Pose indexes
    nose_index: int = 0
    left_shoulder_index: int = 11
    right_shoulder_index: int = 12
    left_elbow_index: int = 13
    right_elbow_index: int = 14
    left_wrist_index: int = 15
    right_wrist_index: int = 16
    pose_scale_epsilon: float = 1e-6

    # Face indexes
    mouth_upper_index: int = 13
    mouth_lower_index: int = 14
    mouth_left_index: int = 61
    mouth_right_index: int = 291
    left_eye_upper_index: int = 159
    left_eye_lower_index: int = 145
    right_eye_upper_index: int = 386
    right_eye_lower_index: int = 374
    left_brow_index: int = 105
    right_brow_index: int = 334
    left_eye_outer_index: int = 33
    right_eye_outer_index: int = 263
    nose_tip_index: int = 1
    face_scale_epsilon: float = 1e-6


def load_phase6_config(path: Path) -> tuple[Phase6Config, dict[str,Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    p = raw.get("pose", {}) or {}
    f = raw.get("face", {}) or {}

    cfg = Phase6Config(
        nose_index=int(p.get("nose_index",0)),
        left_shoulder_index=int(p.get("left_shoulder_index",11)),
        right_shoulder_index=int(p.get("right_shoulder_index",12)),
        left_elbow_index=int(p.get("left_elbow_index",13)),
        right_elbow_index=int(p.get("right_elbow_index",14)),
        left_wrist_index=int(p.get("left_wrist_index",15)),
        right_wrist_index=int(p.get("right_wrist_index",16)),
        pose_scale_epsilon=float(p.get("scale_epsilon",1e-6)),

        mouth_upper_index=int(f.get("mouth_upper_index",13)),
        mouth_lower_index=int(f.get("mouth_lower_index",14)),
        mouth_left_index=int(f.get("mouth_left_index",61)),
        mouth_right_index=int(f.get("mouth_right_index",291)),
        left_eye_upper_index=int(f.get("left_eye_upper_index",159)),
        left_eye_lower_index=int(f.get("left_eye_lower_index",145)),
        right_eye_upper_index=int(f.get("right_eye_upper_index",386)),
        right_eye_lower_index=int(f.get("right_eye_lower_index",374)),
        left_brow_index=int(f.get("left_brow_index",105)),
        right_brow_index=int(f.get("right_brow_index",334)),
        left_eye_outer_index=int(f.get("left_eye_outer_index",33)),
        right_eye_outer_index=int(f.get("right_eye_outer_index",263)),
        nose_tip_index=int(f.get("nose_tip_index",1)),
        face_scale_epsilon=float(f.get("scale_epsilon",1e-6)),
    )

    if cfg.pose_scale_epsilon <= 0 or cfg.face_scale_epsilon <= 0:
        raise ValueError("scale epsilon must be > 0")

    return cfg, raw


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float64)
            - np.asarray(b, dtype=np.float64)
        )
    )


def compute_pose_features(
    pose_xyz: np.ndarray | None,
    pose_present: np.ndarray | None,
    cfg: Phase6Config,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compact upper-body pose features.

    7 normalized landmark positions:
      nose, L/R shoulder, L/R elbow, L/R wrist
      -> 7 * 3 = 21

    plus 6 relation distances:
      L/R wrist -> nose
      L/R wrist -> corresponding shoulder
      L/R wrist -> chest center
      -> 6

    total = 27 dims/frame.

    Coordinates are centered on shoulder midpoint and scaled by shoulder width.
    """
    names = [
        "nose_x","nose_y","nose_z",
        "left_shoulder_x","left_shoulder_y","left_shoulder_z",
        "right_shoulder_x","right_shoulder_y","right_shoulder_z",
        "left_elbow_x","left_elbow_y","left_elbow_z",
        "right_elbow_x","right_elbow_y","right_elbow_z",
        "left_wrist_x","left_wrist_y","left_wrist_z",
        "right_wrist_x","right_wrist_y","right_wrist_z",
        "left_wrist_to_nose",
        "right_wrist_to_nose",
        "left_wrist_to_shoulder",
        "right_wrist_to_shoulder",
        "left_wrist_to_chest",
        "right_wrist_to_chest",
    ]

    if pose_xyz is None:
        return (
            np.zeros((0,27),dtype=np.float32),
            np.zeros((0,),dtype=bool),
            names,
        )

    pose = np.asarray(pose_xyz,dtype=np.float32)
    if pose.ndim != 3 or pose.shape[2] < 3:
        raise ValueError(f"pose_xyz expected (T,N,>=3), got {pose.shape}")

    T = pose.shape[0]

    if pose_present is None:
        present = np.ones(T,dtype=bool)
    else:
        present = np.asarray(pose_present,dtype=bool).reshape(-1)
        if len(present) != T:
            raise ValueError("pose_present length mismatch")

    max_idx = max(
        cfg.nose_index,
        cfg.left_shoulder_index,
        cfg.right_shoulder_index,
        cfg.left_elbow_index,
        cfg.right_elbow_index,
        cfg.left_wrist_index,
        cfg.right_wrist_index,
    )

    if pose.shape[1] <= max_idx:
        raise ValueError(
            f"pose landmark count={pose.shape[1]} insufficient for index {max_idx}"
        )

    out = np.zeros((T,27),dtype=np.float32)
    valid = np.zeros(T,dtype=bool)

    order = [
        cfg.nose_index,
        cfg.left_shoulder_index,
        cfg.right_shoulder_index,
        cfg.left_elbow_index,
        cfg.right_elbow_index,
        cfg.left_wrist_index,
        cfg.right_wrist_index,
    ]

    for t in range(T):
        if not present[t]:
            continue

        frame = pose[t,:,:3]

        required = frame[order]
        if not np.isfinite(required).all():
            continue

        nose = frame[cfg.nose_index]
        ls = frame[cfg.left_shoulder_index]
        rs = frame[cfg.right_shoulder_index]
        le = frame[cfg.left_elbow_index]
        re = frame[cfg.right_elbow_index]
        lw = frame[cfg.left_wrist_index]
        rw = frame[cfg.right_wrist_index]

        shoulder_width = _euclidean(ls,rs)
        if not np.isfinite(shoulder_width) or shoulder_width <= cfg.pose_scale_epsilon:
            continue

        chest = 0.5*(ls+rs)

        # Normalize selected points into shoulder-centered body coordinates.
        pts = np.stack([nose,ls,rs,le,re,lw,rw],axis=0)
        norm = (pts - chest[None,:]) / shoulder_width

        out[t,:21] = norm.reshape(-1).astype(np.float32)

        rel = np.array([
            _euclidean(lw,nose)/shoulder_width,
            _euclidean(rw,nose)/shoulder_width,
            _euclidean(lw,ls)/shoulder_width,
            _euclidean(rw,rs)/shoulder_width,
            _euclidean(lw,chest)/shoulder_width,
            _euclidean(rw,chest)/shoulder_width,
        ],dtype=np.float32)

        if not np.isfinite(rel).all():
            continue

        out[t,21:] = rel
        valid[t] = True

    out[~valid] = 0.0
    return out, valid, names


def compute_face_features(
    face_xyz: np.ndarray | None,
    face_present: np.ndarray | None,
    cfg: Phase6Config,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compact 9-D face expression/orientation proxy:

      0 mouth_open
      1 mouth_width
      2 left_eye_open
      3 right_eye_open
      4 left_brow_height
      5 right_brow_height
      6 head_yaw_proxy
      7 head_pitch_proxy
      8 head_roll_proxy

    Distances are normalized by inter-ocular outer-corner distance.
    Orientation outputs are lightweight proxies, not full PnP head pose.
    """
    names = [
        "mouth_open",
        "mouth_width",
        "left_eye_open",
        "right_eye_open",
        "left_brow_height",
        "right_brow_height",
        "head_yaw_proxy",
        "head_pitch_proxy",
        "head_roll_proxy",
    ]

    if face_xyz is None:
        return (
            np.zeros((0,9),dtype=np.float32),
            np.zeros((0,),dtype=bool),
            names,
        )

    face = np.asarray(face_xyz,dtype=np.float32)
    if face.ndim != 3 or face.shape[2] < 3:
        raise ValueError(f"face_xyz expected (T,N,>=3), got {face.shape}")

    T = face.shape[0]

    if face_present is None:
        present = np.ones(T,dtype=bool)
    else:
        present = np.asarray(face_present,dtype=bool).reshape(-1)
        if len(present) != T:
            raise ValueError("face_present length mismatch")

    required_indexes = [
        cfg.mouth_upper_index,
        cfg.mouth_lower_index,
        cfg.mouth_left_index,
        cfg.mouth_right_index,
        cfg.left_eye_upper_index,
        cfg.left_eye_lower_index,
        cfg.right_eye_upper_index,
        cfg.right_eye_lower_index,
        cfg.left_brow_index,
        cfg.right_brow_index,
        cfg.left_eye_outer_index,
        cfg.right_eye_outer_index,
        cfg.nose_tip_index,
    ]

    max_idx = max(required_indexes)
    if face.shape[1] <= max_idx:
        raise ValueError(
            f"face landmark count={face.shape[1]} insufficient for index {max_idx}"
        )

    out=np.zeros((T,9),dtype=np.float32)
    valid=np.zeros(T,dtype=bool)

    for t in range(T):
        if not present[t]:
            continue

        f = face[t,:,:3]
        req = f[required_indexes]

        if not np.isfinite(req).all():
            continue

        ml = f[cfg.mouth_left_index]
        mr = f[cfg.mouth_right_index]
        mu = f[cfg.mouth_upper_index]
        md = f[cfg.mouth_lower_index]

        leu = f[cfg.left_eye_upper_index]
        lel = f[cfg.left_eye_lower_index]
        reu = f[cfg.right_eye_upper_index]
        rel = f[cfg.right_eye_lower_index]

        lb = f[cfg.left_brow_index]
        rb = f[cfg.right_brow_index]

        leo = f[cfg.left_eye_outer_index]
        reo = f[cfg.right_eye_outer_index]
        nose = f[cfg.nose_tip_index]

        face_scale = _euclidean(leo,reo)
        if not np.isfinite(face_scale) or face_scale <= cfg.face_scale_epsilon:
            continue

        mouth_open = _euclidean(mu,md)/face_scale
        mouth_width = _euclidean(ml,mr)/face_scale

        left_eye_open = _euclidean(leu,lel)/face_scale
        right_eye_open = _euclidean(reu,rel)/face_scale

        left_eye_center = 0.5*(leu+lel)
        right_eye_center = 0.5*(reu+rel)
        left_brow_height = _euclidean(lb,left_eye_center)/face_scale
        right_brow_height = _euclidean(rb,right_eye_center)/face_scale

        eye_mid = 0.5*(leo+reo)

        # Yaw proxy: horizontal nose displacement relative to eye midpoint.
        head_yaw = float((nose[0]-eye_mid[0])/face_scale)

        # Pitch proxy: vertical nose displacement relative to eye midpoint.
        head_pitch = float((nose[1]-eye_mid[1])/face_scale)

        # Roll proxy: slope angle of eye line, normalized by pi.
        eye_vec = reo[:2]-leo[:2]
        roll = float(np.arctan2(eye_vec[1],eye_vec[0]) / np.pi)

        values = np.array([
            mouth_open,
            mouth_width,
            left_eye_open,
            right_eye_open,
            left_brow_height,
            right_brow_height,
            head_yaw,
            head_pitch,
            roll,
        ],dtype=np.float32)

        if not np.isfinite(values).all():
            continue

        out[t]=values
        valid[t]=True

    out[~valid]=0.0
    return out,valid,names


def build_phase6_features(
    *,
    phase5_full417: np.ndarray,
    pose_xyz: np.ndarray | None,
    pose_present: np.ndarray | None,
    face_xyz: np.ndarray | None,
    face_present: np.ndarray | None,
    cfg: Phase6Config,
) -> dict[str,np.ndarray]:
    hands = np.asarray(phase5_full417,dtype=np.float32)
    if hands.ndim != 2 or hands.shape[1] != 417:
        raise ValueError(
            f"phase5_full417 must have shape (T,417), got {hands.shape}"
        )

    T = hands.shape[0]

    pose27,pose_valid,pose_names = compute_pose_features(
        pose_xyz,pose_present,cfg
    )

    face9,face_valid,face_names = compute_face_features(
        face_xyz,face_present,cfg
    )

    # Missing entire modality is represented by zero+mask.
    if pose27.shape[0] == 0:
        pose27 = np.zeros((T,27),dtype=np.float32)
        pose_valid = np.zeros(T,dtype=bool)
    elif pose27.shape[0] != T:
        raise ValueError("pose frame count mismatch with Phase 5")

    if face9.shape[0] == 0:
        face9 = np.zeros((T,9),dtype=np.float32)
        face_valid = np.zeros(T,dtype=bool)
    elif face9.shape[0] != T:
        raise ValueError("face frame count mismatch with Phase 5")

    hands_pose444 = np.concatenate([hands,pose27],axis=1).astype(np.float32)
    hands_pose_face453 = np.concatenate(
        [hands,pose27,face9],axis=1
    ).astype(np.float32)

    return {
        "hands_full417":hands,
        "pose27":pose27,
        "pose_valid":pose_valid.astype(np.bool_),
        "face9":face9,
        "face_valid":face_valid.astype(np.bool_),
        "ablation_hands417":hands,
        "ablation_hands_pose444":hands_pose444,
        "ablation_hands_pose_face453":hands_pose_face453,
    }


def inspect_phase6_arrays(arrays: dict[str,np.ndarray]) -> dict[str,Any]:
    T=arrays["hands_full417"].shape[0]

    expected={
        "hands_full417":(T,417),
        "pose27":(T,27),
        "pose_valid":(T,),
        "face9":(T,9),
        "face_valid":(T,),
        "ablation_hands417":(T,417),
        "ablation_hands_pose444":(T,444),
        "ablation_hands_pose_face453":(T,453),
    }

    issues=[]
    for k,shape in expected.items():
        if k not in arrays:
            issues.append(f"missing_array:{k}")
        elif arrays[k].shape != shape:
            issues.append(f"bad_shape:{k}:{arrays[k].shape}!={shape}")

    for k in (
        "hands_full417",
        "pose27",
        "face9",
        "ablation_hands_pose444",
        "ablation_hands_pose_face453",
    ):
        if k in arrays and not np.isfinite(arrays[k]).all():
            issues.append(f"non_finite:{k}")

    return {
        "frame_count":T,
        "hands_dim":417,
        "pose_dim":27,
        "face_dim":9,
        "hands_pose_dim":444,
        "hands_pose_face_dim":453,
        "pose_valid_ratio":float(arrays["pose_valid"].mean()) if T else 0.0,
        "face_valid_ratio":float(arrays["face_valid"].mean()) if T else 0.0,
        "issues":issues,
        "pass":not issues,
    }
