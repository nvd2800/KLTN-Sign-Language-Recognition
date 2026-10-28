"""Phase08 exploratory confidence calibration for a small or one-signer UNKNOWN set.

This script NEVER writes artifacts/phase8_unknown/calibrated_threshold.json,
which is reserved for the independently validated Phase08 pipeline.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.unknown_phase8 import threshold_metrics, choose_threshold


def rp(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read_mode(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Exploratory marker not found: {path}; run process_phase8_unknown.py --quick-test first")
    info = json.loads(path.read_text(encoding="utf-8"))
    if info.get("mode") != "exploratory_val_only":
        raise ValueError("Not a quick-test dataset. For official evaluation use calibrate_phase8_threshold.py")
    return info


def main():
    parser = argparse.ArgumentParser(description="Phase08 EXPLORATORY small UNKNOWN calibration")
    parser.add_argument("--config", default="configs/phase8_unknown.yaml")
    parser.add_argument("--out", default="artifacts/phase8_unknown_dev")
    args = parser.parse_args()
    cfg = yaml.safe_load(rp(args.config).read_text(encoding="utf-8"))
    marker_path = rp(cfg["unknown_data"]["manifest_csv"]).parent / "unknown_manifest_mode.json"
    mode = read_mode(marker_path)
    unknown_npz = rp(cfg["unknown_data"]["val_npz"])
    known_npz = rp(cfg["known_data"]["val_npz"])
    with np.load(known_npz, allow_pickle=False) as data:
        known_x = np.asarray(data["X"], dtype=np.float32)
    with np.load(unknown_npz, allow_pickle=False) as data:
        unknown_x = np.asarray(data["X"], dtype=np.float32)
    if known_x.ndim != 3 or unknown_x.ndim != 3 or len(known_x) == 0 or len(unknown_x) == 0:
        raise ValueError("KNOWN VAL and UNKNOWN development VAL must be non-empty [N,T,F]")
    if known_x.shape[1:] != unknown_x.shape[1:]:
        raise ValueError(f"Input mismatch: known {known_x.shape}, unknown {unknown_x.shape}")
    model_path = rp(cfg["model"]["experiment_dir"]) / "model.keras"
    if not model_path.is_file():
        model_path = rp(cfg["model"]["fallback_model"])
    if not model_path.is_file():
        raise FileNotFoundError("Missing frozen TCN model; check Phase08 config model paths")
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    known_probs = np.asarray(model.predict(known_x, verbose=0))
    unknown_probs = np.asarray(model.predict(unknown_x, verbose=0))
    if known_probs.ndim != 2 or unknown_probs.ndim != 2 or known_probs.shape[1] != unknown_probs.shape[1]:
        raise ValueError("TCN output probability dimensions do not match")
    known_conf = known_probs.max(axis=1)
    unknown_conf = unknown_probs.max(axis=1)
    sweep = [threshold_metrics(known_conf, unknown_conf, float(t))
             for t in cfg["thresholds"]["candidates"]]
    best = choose_threshold(sweep)
    out = rp(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "threshold_sweep_dev.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = list(asdict(sweep[0]))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([asdict(row) for row in sweep])
    result = {
        "status": "EXPLORATORY_NOT_VALIDATED",
        "threshold_dev_only": best.threshold,
        "metrics_dev_only": asdict(best),
        "known_val_count": int(len(known_x)),
        "unknown_dev_count": int(len(unknown_x)),
        "categories": mode.get("categories", []),
        "subjects": mode.get("subjects", []),
        "warning": "No held-out independent UNKNOWN test. Do NOT report as Phase08 PASS or production calibration.",
    }
    (out / "threshold_dev_only.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nPHASE 08 UNKNOWN — DEVELOPMENT / EXPLORATORY ONLY")
    print("Known VAL:", len(known_x), "UNKNOWN dev:", len(unknown_x))
    print("Observed categories:", ', '.join(mode.get('categories', [])))
    print("WARNING: thresholds from a tiny UNKNOWN set may generalize poorly.")
    for row in sweep:
        print(f"t={row.threshold:.2f} known_accept={row.known_accept_rate:.3f} "
              f"unknown_reject={row.unknown_reject_rate:.3f} "
              f"FAR={row.false_accept_rate:.3f} FRR={row.false_reject_rate:.3f}")
    print("Exploratory threshold:", best.threshold)
    print("Development report:", out / "threshold_dev_only.json")
    print("DO NOT use this threshold to claim final UNKNOWN performance.")


if __name__ == "__main__":
    main()
