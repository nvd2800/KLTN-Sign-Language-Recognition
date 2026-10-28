"""Conservative, mask-aware spatial/handshape gate layered AFTER TCN V2.

Feature extraction remains 453 for TCN. Candidate32 is diagnostic-only.
Thresholds must come from labeled validation clips; DEV profiles never imply a
validated sign-language system. No TensorFlow/OpenCV dependency.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import json
import numpy as np

GROUPS = {
    "hand_body_position12": (0, 12, "wrong_position"),
    "finger_curl10": (12, 22, "wrong_handshape"),
    "palm_orientation6": (22, 28, "wrong_orientation"),
}
PROFILE_VERSION = 1


def fix_length(x: np.ndarray, n: int = 32) -> np.ndarray:
    x = np.asarray(x)
    if not len(x):
        raise ValueError("Empty feature sequence")
    return x[np.rint(np.linspace(0, len(x)-1, n)).astype(int)]


def _validate(x, m):
    x = np.asarray(x, np.float32)
    m = np.asarray(m, bool)
    if x.ndim != 2 or x.shape[1] != 32 or m.shape != x.shape:
        raise ValueError(f"Expected features/mask [T,32], got {x.shape}/{m.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite candidate features")
    return fix_length(x), fix_length(m)


def masked_group_distance(sample, sample_mask, reference, reference_mask,
                          group, *, min_overlap=0.55, min_valid=8):
    """Masked RMS aligned to 32 frames; require substantial common reference support.

    The denominator is *reference valid entries*, not 32*group_size: a valid
    one-hand sign is NOT penalized for the other hand being absent.
    """
    x, mx = _validate(sample, sample_mask)
    r, mr = _validate(reference, reference_mask)
    if group not in GROUPS:
        raise ValueError(f"Unknown group: {group}")
    lo, hi, _ = GROUPS[group]
    common = mx[:,lo:hi] & mr[:,lo:hi]
    ref_support = int(mr[:,lo:hi].sum())
    sample_support = int(mx[:,lo:hi].sum())
    n = int(common.sum())
    coverage = n / ref_support if ref_support else 0.0
    if n < min_valid or coverage < min_overlap or sample_support < min_valid:
        return {"distance": None, "compared": n,
                "ref_valid": ref_support, "candidate_valid": sample_support,
                "relative_coverage": round(coverage, 6)}
    diff = x[:,lo:hi][common].astype(np.float64) - r[:,lo:hi][common].astype(np.float64)
    return {"distance": float(np.sqrt(np.mean(diff*diff))),
            "compared": n, "ref_valid": ref_support,
            "candidate_valid": sample_support, "relative_coverage": round(coverage, 6)}


def min_gallery_distances(x, mask, references, groups, *, min_overlap=0.55):
    out = {}
    for group in groups:
        results = [masked_group_distance(x, mask, a, m, group,
                  min_overlap=min_overlap) for a, m in references]
        usable = [r for r in results if r["distance"] is not None]
        if usable:
            out[group] = min(usable, key=lambda r: r["distance"])
        else:
            out[group] = max(results, key=lambda r: r["compared"]) if results else {
                "distance": None, "relative_coverage": 0.0, "compared": 0}
    return out


def threshold_on_validation(positive_scores, negative_scores):
    """Choose only if the two classes separate beyond measurement precision.

    Max balanced validation recall/rejection. Cannot manufacture separation.
    """
    pos = np.asarray(positive_scores, float)
    neg = np.asarray(negative_scores, float)
    if not len(pos) or not len(neg) or not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError("Both positive and negative validation distances required")
    vals = sorted(set(pos.tolist() + neg.tolist()))
    eps = 1e-6
    candidates = [max(0., vals[0]-eps)] + [(a+b)/2 for a,b in zip(vals[:-1], vals[1:])] + [vals[-1]+eps]
    scored = []
    for thr in candidates:
        accept = float(np.mean(pos <= thr))
        reject = float(np.mean(neg > thr))
        scored.append((0.5*(accept+reject),accept,reject,thr))
    scored.sort(key=lambda row:(-row[0], -row[1], row[3]))
    balance, accept, reject, threshold = scored[0]
    # At random-level balance, the modality does not reliably discriminate.
    if balance <= 0.5 + 1e-6:
        return None, {"balanced":balance,"positive_accept":accept,"negative_reject":reject,
                      "usable":False,"reason":"NO_DISCRIMINATION"}
    return float(threshold), {"balanced":balance,"positive_accept":accept,
                              "negative_reject":reject,"usable":True}


@dataclass(frozen=True)
class GateResult:
    decision: str              # PASS | REJECT | INSUFFICIENT | BYPASS_UNCONFIGURED | SKIP
    reason: str
    scores: dict
    thresholds: dict


def validate_profile(profile: Mapping):
    if profile.get("schema_version") != PROFILE_VERSION:
        raise ValueError("Unsupported spatial gate profile version")
    if profile.get("calibration_mode") not in ("DEV_ONLY", "VALIDATED"):
        raise ValueError("Profile must declare calibration_mode")
    if not isinstance(profile.get("classes"), dict) or not profile["classes"]:
        raise ValueError("Profile has no class references")
    for label, spec in profile["classes"].items():
        if label in ("UNKNOWN", "REST"):
            raise ValueError("Do not spatial-gate UNKNOWN/REST")
        refs = spec.get("references", [])
        thresholds = spec.get("thresholds", {})
        if not refs or not thresholds:
            raise ValueError(f"Missing references/thresholds for {label}")
        for ref in refs:
            _validate(ref["features"], ref["mask"])
        for group, value in thresholds.items():
            if group not in GROUPS or not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid threshold {group} = {value}")
    return True


def load_profile(path: Path):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_profile(p)
    return p


def evaluate(profile, label: str, features, mask) -> GateResult:
    if label in ("REST", "UNKNOWN"):
        return GateResult("SKIP", "NON_KNOWN", {}, {})
    if label not in profile["classes"]:
        # Explicitly marked as NOT spatially protected. This avoids silently
        # pretending to validate other classes when only xin_chao is calibrated.
        return GateResult("BYPASS_UNCONFIGURED", "CLASS_NOT_CALIBRATED", {}, {})
    spec = profile["classes"][label]
    references = [(np.asarray(ref["features"],np.float32), np.asarray(ref["mask"],bool))
                  for ref in spec["references"]]
    thresholds = spec["thresholds"]
    distances = min_gallery_distances(features, mask, references, thresholds.keys(),
                                      min_overlap=float(profile.get("min_overlap", 0.55)))
    missing = [g for g in thresholds if distances[g]["distance"] is None]
    if missing:
        return GateResult("INSUFFICIENT", "MISSING_VALID_FEATURES:"+",".join(missing), distances, dict(thresholds))
    failed = [g for g in thresholds if distances[g]["distance"] > float(thresholds[g])]
    if failed:
        return GateResult("REJECT", "FEATURE_MISMATCH:"+",".join(failed), distances, dict(thresholds))
    return GateResult("PASS", "SPATIAL_HANDSHAPE_MATCH", distances, dict(thresholds))


def trial_from_npz(path: Path):
    with np.load(path, allow_pickle=False) as d:
        return _validate(d["candidate_v4_32"], d["candidate_v4_mask32"])
