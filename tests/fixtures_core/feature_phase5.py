from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import yaml


# MediaPipe hand landmark chains.
FINGER_CHAINS = {
    "thumb":  [0, 1, 2, 3, 4],
    "index":  [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring":   [0, 13, 14, 15, 16],
    "pinky":  [0, 17, 18, 19, 20],
}

# Three angles / finger:
# at CMC/MCP, PIP/IP, DIP respectively.
ANGLE_TRIPLES = [
    (0,1,2),(1,2,3),(2,3,4),       # thumb
    (0,5,6),(5,6,7),(6,7,8),       # index
    (0,9,10),(9,10,11),(10,11,12), # middle
    (0,13,14),(13,14,15),(14,15,16),# ring
    (0,17,18),(17,18,19),(18,19,20),# pinky
]


@dataclass(frozen=True)
class Phase5Config:
    angle_epsilon: float = 1e-8
    normalize_angles_by_pi: bool = True

    distance_epsilon: float = 1e-6
    pose_nose_index: int = 0
    pose_left_shoulder_index: int = 11
    pose_right_shoulder_index: int = 12

    use_time_delta: bool = False
    velocity_clip_abs: float = 5.0
    acceleration_clip_abs: float = 8.0


def load_phase5_config(path: Path) -> tuple[Phase5Config, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    angles = raw.get("angles", {}) or {}
    dist = raw.get("distances", {}) or {}
    motion = raw.get("motion", {}) or {}

    cfg = Phase5Config(
        angle_epsilon=float(angles.get("epsilon", 1e-8)),
        normalize_angles_by_pi=bool(angles.get("normalize_by_pi", True)),
        distance_epsilon=float(dist.get("epsilon", 1e-6)),
        pose_nose_index=int(dist.get("pose_nose_index", 0)),
        pose_left_shoulder_index=int(dist.get("pose_left_shoulder_index", 11)),
        pose_right_shoulder_index=int(dist.get("pose_right_shoulder_index", 12)),
        use_time_delta=bool(motion.get("use_time_delta", False)),
        velocity_clip_abs=float(motion.get("velocity_clip_abs", 5.0)),
        acceleration_clip_abs=float(motion.get("acceleration_clip_abs", 8.0)),
    )

    if cfg.angle_epsilon <= 0 or cfg.distance_epsilon <= 0:
        raise ValueError("epsilons must be > 0")
    if cfg.velocity_clip_abs <= 0 or cfg.acceleration_clip_abs <= 0:
        raise ValueError("clip values must be > 0")

    return cfg, raw


def angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float) -> float:
    """Angle ABC in radians, robust to near-zero vectors."""
    u = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)

    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))

    if not np.isfinite(nu) or not np.isfinite(nv) or nu <= eps or nv <= eps:
        return 0.0

    cosine = float(np.dot(u, v) / (nu * nv))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.arccos(cosine))


def compute_hand_angles(
    hand_xyz: np.ndarray,
    present: np.ndarray,
    *,
    eps: float,
    normalize_by_pi: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      angles: (T, 15)
      mask:   (T, 15)
    """
    hand_xyz = np.asarray(hand_xyz, dtype=np.float32)
    present = np.asarray(present, dtype=bool).reshape(-1)

    T = hand_xyz.shape[0]
    if hand_xyz.shape != (T, 21, 3):
        raise ValueError(f"hand_xyz bad shape: {hand_xyz.shape}")
    if len(present) != T:
        raise ValueError("presence length mismatch")

    out = np.zeros((T, 15), dtype=np.float32)
    mask = np.zeros((T, 15), dtype=bool)

    for t in range(T):
        if not present[t]:
            continue
        if not np.isfinite(hand_xyz[t]).all():
            continue

        for j, (ia, ib, ic) in enumerate(ANGLE_TRIPLES):
            theta = angle_abc(
                hand_xyz[t, ia],
                hand_xyz[t, ib],
                hand_xyz[t, ic],
                eps,
            )

            # If vectors were degenerate angle_abc returns 0.0.
            # A true 0-degree anatomical angle is unlikely; we keep mask false
            # only when either adjacent segment is near zero.
            u = hand_xyz[t, ia] - hand_xyz[t, ib]
            v = hand_xyz[t, ic] - hand_xyz[t, ib]
            valid = (
                np.isfinite(u).all()
                and np.isfinite(v).all()
                and np.linalg.norm(u) > eps
                and np.linalg.norm(v) > eps
            )

            if valid:
                out[t, j] = theta / np.pi if normalize_by_pi else theta
                mask[t, j] = True

    return out, mask


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def compute_distance_features(
    *,
    left_norm: np.ndarray,
    right_norm: np.ndarray,
    left_present: np.ndarray,
    right_present: np.ndarray,
    raw_left: np.ndarray | None,
    raw_right: np.ndarray | None,
    scale2: np.ndarray,
    pose_xyz: np.ndarray | None,
    pose_present: np.ndarray | None,
    cfg: Phase5Config,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Distance feature order (9):
      0 left_thumb_index
      1 right_thumb_index
      2 left_right_wrist
      3 left_hand_face
      4 right_hand_face
      5 left_hand_shoulder
      6 right_hand_shoulder
      7 left_hand_chest
      8 right_hand_chest

    Body-relative distances are optional. Missing pose => zero + mask false.
    """
    T = left_norm.shape[0]
    out = np.zeros((T, 9), dtype=np.float32)
    mask = np.zeros((T, 9), dtype=bool)

    names = [
        "left_thumb_index",
        "right_thumb_index",
        "left_right_wrist",
        "left_hand_face",
        "right_hand_face",
        "left_hand_shoulder",
        "right_hand_shoulder",
        "left_hand_chest",
        "right_hand_chest",
    ]

    lp = np.asarray(left_present, dtype=bool)
    rp = np.asarray(right_present, dtype=bool)

    # 1) Thumb tip (4) ↔ Index tip (8), already scale-normalized per hand.
    for t in range(T):
        if lp[t]:
            out[t,0] = _euclidean(left_norm[t,4], left_norm[t,8])
            mask[t,0] = np.isfinite(out[t,0])

        if rp[t]:
            out[t,1] = _euclidean(right_norm[t,4], right_norm[t,8])
            mask[t,1] = np.isfinite(out[t,1])

    # 2) Left wrist ↔ Right wrist.
    # Phase-4 local normalized wrists are both zero, so use raw wrist positions
    # and divide by mean valid hand scale.
    if raw_left is not None and raw_right is not None:
        for t in range(T):
            if not (lp[t] and rp[t]):
                continue

            s = np.asarray(scale2[t], dtype=np.float64)
            valid_scales = s[np.isfinite(s) & (s > cfg.distance_epsilon)]

            if len(valid_scales) == 0:
                continue

            ref_scale = float(np.mean(valid_scales))

            d = _euclidean(raw_left[t,0], raw_right[t,0]) / ref_scale

            if np.isfinite(d):
                out[t,2] = d
                mask[t,2] = True

    # 3) Body-relative distances using Pose if available.
    # Normalize by shoulder width for camera-distance robustness.
    if (
        pose_xyz is not None
        and pose_present is not None
        and raw_left is not None
        and raw_right is not None
    ):
        pose_xyz = np.asarray(pose_xyz, dtype=np.float32)
        pose_present = np.asarray(pose_present, dtype=bool).reshape(-1)

        if pose_xyz.ndim == 3 and pose_xyz.shape[0] == T and pose_xyz.shape[2] >= 3:
            max_idx = max(
                cfg.pose_nose_index,
                cfg.pose_left_shoulder_index,
                cfg.pose_right_shoulder_index,
            )

            if pose_xyz.shape[1] > max_idx:
                for t in range(T):
                    if not pose_present[t] or not np.isfinite(pose_xyz[t]).all():
                        continue

                    nose = pose_xyz[t,cfg.pose_nose_index,:3]
                    ls = pose_xyz[t,cfg.pose_left_shoulder_index,:3]
                    rs = pose_xyz[t,cfg.pose_right_shoulder_index,:3]
                    shoulder_width = _euclidean(ls, rs)

                    if not np.isfinite(shoulder_width) or shoulder_width <= cfg.distance_epsilon:
                        continue

                    chest = 0.5 * (ls + rs)

                    if lp[t]:
                        lw = raw_left[t,0,:3]
                        vals = [
                            _euclidean(lw, nose) / shoulder_width,
                            _euclidean(lw, ls) / shoulder_width,
                            _euclidean(lw, chest) / shoulder_width,
                        ]
                        for idx, val in zip((3,5,7), vals):
                            if np.isfinite(val):
                                out[t,idx] = val
                                mask[t,idx] = True

                    if rp[t]:
                        rw = raw_right[t,0,:3]
                        vals = [
                            _euclidean(rw, nose) / shoulder_width,
                            _euclidean(rw, rs) / shoulder_width,
                            _euclidean(rw, chest) / shoulder_width,
                        ]
                        for idx, val in zip((4,6,8), vals):
                            if np.isfinite(val):
                                out[t,idx] = val
                                mask[t,idx] = True

    out[~mask] = 0.0
    return out, mask, names


def compute_velocity_acceleration(
    position126: np.ndarray,
    hand_presence2: np.ndarray,
    timestamps: np.ndarray,
    cfg: Phase5Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Missing-aware motion.

    Velocity for a hand is valid only if that same hand exists at t-1 and t.
    Acceleration is valid only when two consecutive velocity steps are valid.

    Default roadmap definition:
      v_t = X_t - X_(t-1)
      a_t = v_t - v_(t-1)

    If use_time_delta=true, differences are divided by Δt.
    """
    X = np.asarray(position126, dtype=np.float32)
    p = np.asarray(hand_presence2, dtype=bool)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)

    T = X.shape[0]
    if X.shape != (T,126):
        raise ValueError(f"position shape must be (T,126), got {X.shape}")
    if p.shape != (T,2):
        raise ValueError(f"hand_presence2 must be (T,2), got {p.shape}")
    if len(ts) != T:
        raise ValueError("timestamps length mismatch")

    V = np.zeros_like(X, dtype=np.float32)
    A = np.zeros_like(X, dtype=np.float32)
    vmask = np.zeros((T,2), dtype=bool)
    amask = np.zeros((T,2), dtype=bool)

    for t in range(1,T):
        dt = float(ts[t]-ts[t-1])
        if cfg.use_time_delta and (not np.isfinite(dt) or dt <= 0):
            continue

        for h, sl in enumerate((slice(0,63), slice(63,126))):
            if p[t-1,h] and p[t,h]:
                diff = X[t,sl] - X[t-1,sl]
                if cfg.use_time_delta:
                    diff = diff / dt

                diff = np.clip(
                    diff,
                    -cfg.velocity_clip_abs,
                    cfg.velocity_clip_abs,
                )

                if np.isfinite(diff).all():
                    V[t,sl] = diff
                    vmask[t,h] = True

    for t in range(2,T):
        dt = float(ts[t]-ts[t-1])
        if cfg.use_time_delta and (not np.isfinite(dt) or dt <= 0):
            continue

        for h, sl in enumerate((slice(0,63), slice(63,126))):
            if vmask[t-1,h] and vmask[t,h]:
                diff = V[t,sl] - V[t-1,sl]
                if cfg.use_time_delta:
                    diff = diff / dt

                diff = np.clip(
                    diff,
                    -cfg.acceleration_clip_abs,
                    cfg.acceleration_clip_abs,
                )

                if np.isfinite(diff).all():
                    A[t,sl] = diff
                    amask[t,h] = True

    return V, A, vmask, amask


def build_phase5_features(
    *,
    phase4_arrays: dict[str,np.ndarray],
    raw_left: np.ndarray | None,
    raw_right: np.ndarray | None,
    pose_xyz: np.ndarray | None,
    pose_present: np.ndarray | None,
    cfg: Phase5Config,
) -> dict[str,np.ndarray]:
    pos = np.asarray(phase4_arrays["normalized_xyz126"], dtype=np.float32)
    left = np.asarray(phase4_arrays["left_normalized_xyz"], dtype=np.float32)
    right = np.asarray(phase4_arrays["right_normalized_xyz"], dtype=np.float32)
    hp = np.asarray(phase4_arrays["hand_presence2"], dtype=bool)
    scale2 = np.asarray(phase4_arrays["scale2"], dtype=np.float32)
    ts = np.asarray(phase4_arrays["timestamps"], dtype=np.float64)

    la, lam = compute_hand_angles(
        left, hp[:,0],
        eps=cfg.angle_epsilon,
        normalize_by_pi=cfg.normalize_angles_by_pi,
    )
    ra, ram = compute_hand_angles(
        right, hp[:,1],
        eps=cfg.angle_epsilon,
        normalize_by_pi=cfg.normalize_angles_by_pi,
    )

    angles30 = np.concatenate([la,ra], axis=1).astype(np.float32)
    angle_mask30 = np.concatenate([lam,ram], axis=1).astype(np.bool_)

    dist9, dist_mask9, _ = compute_distance_features(
        left_norm=left,
        right_norm=right,
        left_present=hp[:,0],
        right_present=hp[:,1],
        raw_left=raw_left,
        raw_right=raw_right,
        scale2=scale2,
        pose_xyz=pose_xyz,
        pose_present=pose_present,
        cfg=cfg,
    )

    vel126, acc126, velmask2, accmask2 = compute_velocity_acceleration(
        pos, hp, ts, cfg
    )

    # Ablation-ready sets.
    p126 = pos
    pv252 = np.concatenate([pos,vel126],axis=1).astype(np.float32)
    pva378 = np.concatenate([pos,vel126,acc126],axis=1).astype(np.float32)
    full417 = np.concatenate(
        [pos,angles30,dist9,vel126,acc126],
        axis=1,
    ).astype(np.float32)

    return {
        "position126":p126,
        "angles30":angles30,
        "angle_mask30":angle_mask30,
        "distances9":dist9,
        "distance_mask9":dist_mask9,
        "velocity126":vel126,
        "acceleration126":acc126,
        "velocity_valid2":velmask2,
        "acceleration_valid2":accmask2,
        "hand_presence2":hp.astype(np.bool_),
        "timestamps":ts,
        "ablation_position126":p126,
        "ablation_position_velocity252":pv252,
        "ablation_position_velocity_acceleration378":pva378,
        "full_features417":full417,
    }


def inspect_phase5_arrays(arrays: dict[str,np.ndarray]) -> dict[str,Any]:
    T = arrays["position126"].shape[0]
    expected = {
        "position126":(T,126),
        "angles30":(T,30),
        "angle_mask30":(T,30),
        "distances9":(T,9),
        "distance_mask9":(T,9),
        "velocity126":(T,126),
        "acceleration126":(T,126),
        "velocity_valid2":(T,2),
        "acceleration_valid2":(T,2),
        "ablation_position126":(T,126),
        "ablation_position_velocity252":(T,252),
        "ablation_position_velocity_acceleration378":(T,378),
        "full_features417":(T,417),
    }

    issues=[]
    for k,shape in expected.items():
        if k not in arrays:
            issues.append(f"missing_array:{k}")
        elif arrays[k].shape != shape:
            issues.append(f"bad_shape:{k}:{arrays[k].shape}!={shape}")

    for k in (
        "position126","angles30","distances9","velocity126",
        "acceleration126","full_features417"
    ):
        if k in arrays and not np.isfinite(arrays[k]).all():
            issues.append(f"non_finite:{k}")

    return {
        "frame_count":T,
        "full_feature_dim":417,
        "angle_dim":30,
        "distance_dim":9,
        "velocity_dim":126,
        "acceleration_dim":126,
        "angle_valid_ratio":float(arrays["angle_mask30"].mean()) if T else 0.0,
        "distance_valid_ratio":float(arrays["distance_mask9"].mean()) if T else 0.0,
        "velocity_valid_ratio":float(arrays["velocity_valid2"].mean()) if T else 0.0,
        "acceleration_valid_ratio":float(arrays["acceleration_valid2"].mean()) if T else 0.0,
        "issues":issues,
        "pass":not issues,
    }
