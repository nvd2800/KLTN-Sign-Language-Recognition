from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json

import numpy as np
import yaml


FEATURE_ARRAYS = {
    "hands417": ("ablation_hands417", 417),
    "hands_pose444": ("ablation_hands_pose444", 444),
    "hands_pose_face453": ("ablation_hands_pose_face453", 453),
}


@dataclass(frozen=True)
class Phase7Config:
    target_frames: int = 32
    short_sequence_policy: str = "pad"
    standardize_epsilon: float = 1e-6
    active_feature: str = "hands_pose_face453"


def load_phase7_config(path: Path) -> tuple[Phase7Config, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    seq = raw.get("sequence", {}) or {}
    feat = raw.get("feature", {}) or {}

    cfg = Phase7Config(
        target_frames=int(seq.get("target_frames", 32)),
        short_sequence_policy=str(seq.get("short_sequence_policy", "pad")),
        standardize_epsilon=float(seq.get("standardize_epsilon", 1e-6)),
        active_feature=str(feat.get("active", "hands_pose_face453")),
    )

    if cfg.target_frames <= 1:
        raise ValueError("target_frames must be > 1")
    if cfg.short_sequence_policy not in {"pad"}:
        raise ValueError("short_sequence_policy currently supports only 'pad'")
    if cfg.standardize_epsilon <= 0:
        raise ValueError("standardize_epsilon must be > 0")
    if cfg.active_feature not in FEATURE_ARRAYS:
        raise ValueError(
            f"Unknown feature set {cfg.active_feature}; "
            f"choose from {sorted(FEATURE_ARRAYS)}"
        )

    return cfg, raw


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def uniform_resample_or_pad(
    sequence: np.ndarray,
    target_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize temporal length without inventing labels.

    - If T >= target_frames:
        sample target_frames uniformly from the original sequence.
        All output timesteps are valid.
    - If T < target_frames:
        preserve all original frames, then zero-pad.
        temporal_mask marks only real frames as valid.

    Returns:
        X_fixed      (target_frames, F)
        temporal_mask(target_frames,)
        source_index (target_frames,), -1 for padded frames
    """
    x = np.asarray(sequence, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"sequence must be 2-D (T,F), got {x.shape}")

    T, F = x.shape

    if T <= 0:
        raise ValueError("empty sequence is not allowed")

    out = np.zeros((target_frames, F), dtype=np.float32)
    mask = np.zeros((target_frames,), dtype=bool)
    src = np.full((target_frames,), -1, dtype=np.int32)

    if T >= target_frames:
        idx = np.rint(
            np.linspace(0, T - 1, target_frames)
        ).astype(np.int32)

        # With T >= target, linspace/rint should be monotonic and practically
        # unique; enforce monotonic non-decreasing for robustness.
        idx = np.maximum.accumulate(idx)
        idx = np.clip(idx, 0, T - 1)

        out[:] = x[idx]
        mask[:] = True
        src[:] = idx
        return out, mask, src

    # Short sequence: keep original order and pad at the end.
    out[:T] = x
    mask[:T] = True
    src[:T] = np.arange(T, dtype=np.int32)

    return out, mask, src


def fit_standardizer_train_only(
    X_train: np.ndarray,
    mask_train: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit mean/std ONLY from valid TRAIN timesteps.

    X_train:   (N, T, F)
    mask_train:(N, T)
    """
    X = np.asarray(X_train, dtype=np.float32)
    M = np.asarray(mask_train, dtype=bool)

    if X.ndim != 3:
        raise ValueError(f"X_train expected 3-D, got {X.shape}")
    if M.shape != X.shape[:2]:
        raise ValueError(f"mask shape {M.shape} != {X.shape[:2]}")

    valid = X[M]

    if valid.size == 0:
        raise ValueError("No valid training timesteps to fit standardizer")

    if not np.isfinite(valid).all():
        raise ValueError("Non-finite value found in training features")

    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32)

    std = np.where(std > eps, std, 1.0).astype(np.float32)

    return mean, std


def apply_standardizer(
    X: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Standardize valid timesteps only; padded timesteps remain exact zero.
    This allows Keras Masking(mask_value=0.0) in the GRU.
    """
    X = np.asarray(X, dtype=np.float32)
    M = np.asarray(mask, dtype=bool)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    out = np.zeros_like(X, dtype=np.float32)

    if X.shape[:2] != M.shape:
        raise ValueError("mask shape mismatch")

    if X.shape[-1] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError("standardizer dimension mismatch")

    out[M] = (X[M] - mean) / std

    if not np.isfinite(out).all():
        raise ValueError("Standardization produced NaN/Inf")

    out[~M] = 0.0
    return out


def label_mapping_from_rows(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, int]]:
    labels = sorted({(r.get("sign") or "").strip() for r in rows if (r.get("sign") or "").strip()})

    if not labels:
        raise ValueError("No labels found")

    return labels, {name: idx for idx, name in enumerate(labels)}


def encode_labels(rows: list[dict[str, str]], label_to_id: dict[str, int]) -> np.ndarray:
    values = []
    for r in rows:
        sign = (r.get("sign") or "").strip()
        if sign not in label_to_id:
            raise ValueError(f"Unknown label: {sign}")
        values.append(label_to_id[sign])
    return np.asarray(values, dtype=np.int64)


def split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out = {"train": [], "val": [], "test": []}

    for r in rows:
        split = (r.get("split") or "").strip().lower()

        if split not in out:
            raise ValueError(
                f"Invalid split '{split}' for sample {r.get('sample_id')}"
            )

        out[split].append(r)

    return out


def validate_split_subject_leakage(
    split_map: dict[str, list[dict[str, str]]]
) -> dict[str, Any]:
    subjects = {
        split: {
            (r.get("subject_id") or "").strip()
            for r in rows
            if (r.get("subject_id") or "").strip()
        }
        for split, rows in split_map.items()
    }

    leakage = {
        "train_val": sorted(subjects["train"] & subjects["val"]),
        "train_test": sorted(subjects["train"] & subjects["test"]),
        "val_test": sorted(subjects["val"] & subjects["test"]),
    }

    leakage_count = sum(len(v) for v in leakage.values())

    return {
        "subjects": {k: sorted(v) for k, v in subjects.items()},
        "leakage": leakage,
        "leakage_count": leakage_count,
        "pass": leakage_count == 0,
    }


def load_feature_sequence(
    row: dict[str, str],
    project_root: Path,
    active_feature: str,
) -> np.ndarray:
    array_name, expected_dim = FEATURE_ARRAYS[active_feature]

    p = Path((row.get("features_path") or "").strip())

    if not p.is_absolute():
        p = project_root / p

    if not p.is_file():
        raise FileNotFoundError(f"Phase-6 feature file missing: {p}")

    with np.load(p, allow_pickle=False) as d:
        if array_name not in d.files:
            raise KeyError(f"{array_name} not found in {p}")
        x = np.asarray(d[array_name], dtype=np.float32)

    if x.ndim != 2 or x.shape[1] != expected_dim:
        raise ValueError(
            f"{row.get('sample_id')}: {array_name} shape {x.shape}, "
            f"expected (T,{expected_dim})"
        )

    if not np.isfinite(x).all():
        raise ValueError(f"{row.get('sample_id')}: non-finite input features")

    return x


def build_fixed_split(
    rows: list[dict[str, str]],
    project_root: Path,
    cfg: Phase7Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    sequences = []
    masks = []
    source_indices = []
    sample_ids = []

    expected_dim = FEATURE_ARRAYS[cfg.active_feature][1]

    for row in rows:
        x = load_feature_sequence(
            row=row,
            project_root=project_root,
            active_feature=cfg.active_feature,
        )

        fixed, mask, src = uniform_resample_or_pad(
            x,
            cfg.target_frames,
        )

        if fixed.shape != (cfg.target_frames, expected_dim):
            raise ValueError(f"unexpected fixed shape: {fixed.shape}")

        sequences.append(fixed)
        masks.append(mask)
        source_indices.append(src)
        sample_ids.append((row.get("sample_id") or "").strip())

    if not sequences:
        return (
            np.zeros((0, cfg.target_frames, expected_dim), dtype=np.float32),
            np.zeros((0, cfg.target_frames), dtype=bool),
            np.zeros((0, cfg.target_frames), dtype=np.int32),
            [],
        )

    return (
        np.stack(sequences).astype(np.float32),
        np.stack(masks).astype(np.bool_),
        np.stack(source_indices).astype(np.int32),
        sample_ids,
    )
