"""Validate V3.1 realtime artifacts/config without accessing webcam.

Use: python scripts/check_phase09_v3_1.py [--config configs/phase9_v3_1_spatial.yaml]
     python scripts/check_phase09_v3_1.py --check-model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LABELS = {"REST", "cam_on", "xin_chao", "xin_loi", "UNKNOWN"}


def resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def preflight(root: Path, config: Path, *, check_model: bool = False, dataset: Path | None = None) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []
    info: dict = {}
    if not config.is_file():
        return [f"Config not found: {config}"], [], info
    try:
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        paths = cfg["artifacts"]
        model, stats, labels_file = [resolve(root, paths[k]) for k in ("model", "stats", "labels")]
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as e:
        return [f"Invalid config: {e}"], [], info

    info.update(config=str(config), model=str(model), stats=str(stats), labels=str(labels_file))
    for name, file in (("model", model), ("stats", stats), ("labels", labels_file)):
        if not file.is_file():
            errors.append(f"Missing {name}: {file}")
    if len({model.parent.resolve(), stats.parent.resolve(), labels_file.parent.resolve()}) != 1:
        errors.append("Model / scaler / labels must be from ONE experiment folder")
    if "tcn_v3_spatial" == model.parent.name:
        errors.append("Config points to V3 BASELINE, not V3.1! Select tcn_v3_1_spatial checkpoint")

    if labels_file.is_file():
        try:
            obj = json.loads(labels_file.read_text(encoding="utf-8"))
            labels = obj["labels"] if isinstance(obj, dict) else obj
            info["label_order"] = labels
            if not isinstance(labels, list) or len(labels) != 5 or set(labels) != EXPECTED_LABELS:
                errors.append(f"Unexpected labels: {labels}")
        except (ValueError, TypeError, KeyError) as e:
            errors.append(f"Cannot read labels: {e}")

    if stats.is_file():
        try:
            with np.load(stats, allow_pickle=False) as data:
                mean = np.asarray(data["mean"])
                std = np.asarray(data["std"])
                target = int(np.asarray(data["target_frames"]).item())
                dim = int(np.asarray(data["feature_dim"]).item())
            info.update(input_shape=[target, dim], scaler_shape=list(mean.shape))
            if mean.shape != (513,) or std.shape != (513,) or target != 32 or dim != 513:
                errors.append(f"Expected (32,513) and scaler 513D; got ({target},{dim}), {mean.shape}/{std.shape}")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                errors.append("Scaler contains invalid mean/std values")
        except (KeyError, ValueError, OSError, TypeError) as e:
            errors.append(f"Cannot read scaler: {e}")

    summary_path = model.parent / "dataset_summary.json"
    if summary_path.is_file():
        try:
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            counts = {name: int(s["counts"][name]["total"]) for name in ("train", "val", "test")}
            info.update(checkpoint_dataset_counts=counts, checkpoint_dataset_mode=s.get("mode"))
            if s.get("mode") != "strict":
                warnings.append("Checkpoint summary is not STRICT. Treat as development only")
        except (KeyError, TypeError, ValueError) as e:
            warnings.append(f"Cannot verify checkpoint dataset summary: {e}")
    else:
        warnings.append("No dataset_summary.json next to checkpoint; training provenance not verifiable")

    if dataset is not None:
        summary = dataset / "dataset_summary.json"
        if not summary.is_file():
            warnings.append(f"Comparison dataset summary not found: {summary}")
        else:
            try:
                s = json.loads(summary.read_text(encoding="utf-8"))
                desired = {name:int(s["counts"][name]["total"]) for name in ("train","val","test")}
                info["selected_dataset_counts"] = desired
                if info.get("checkpoint_dataset_counts") is not None and info["checkpoint_dataset_counts"] != desired:
                    errors.append("Checkpoint counts do NOT match selected V3.1 dataset; likely OLD model. Retrain/check paths before reporting results")
            except (KeyError, TypeError, ValueError) as e:
                warnings.append(f"Cannot read comparison dataset: {e}")

    # Optional: loading TensorFlow can be slow, so require --check-model explicitly.
    if check_model and model.is_file():
        try:
            import tensorflow as tf
            loaded = tf.keras.models.load_model(str(model), compile=False)
            input_shape = loaded.input_shape[0] if isinstance(loaded.input_shape, list) else loaded.input_shape
            output_shape = loaded.output_shape[0] if isinstance(loaded.output_shape, list) else loaded.output_shape
            info["keras_input_shape"] = list(input_shape)
            info["keras_output_shape"] = list(output_shape)
            if tuple(input_shape[1:]) != (32, 513) or int(output_shape[-1]) != 5:
                errors.append(f"Model shape incorrect: {input_shape} -> {output_shape}")
        except Exception as e:
            errors.append(f"Could not load TensorFlow model: {type(e).__name__}: {e}")

    if not cfg.get("logging", {}).get("predictions", "").find("v3_1") >= 0:
        warnings.append("Realtime log filename does not mention v3_1; may mix V3 & V3.1 logs")
    return errors, warnings, info


def main() -> int:
    p = argparse.ArgumentParser(description="TCN V3.1 realtime model/scaler/labels preflight")
    p.add_argument("--config", default="configs/phase9_v3_1_spatial.yaml")
    p.add_argument("--check-model", action="store_true")
    p.add_argument("--dataset", default="dataset/processed/temporal_v3_1_spatial")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()
    e, w, info = preflight(ROOT, resolve(ROOT, args.config), check_model=args.check_model, dataset=resolve(ROOT,args.dataset))
    print("=== PHASE 09 / TCN V3.1 PRE-FLIGHT ===")
    for k,v in info.items(): print(f"{k}: {v}")
    for item in w: print("WARNING:",item)
    for item in e: print("ERROR:",item)
    print("VERDICT:", "PASS" if not e else "BLOCKED")
    if args.json_out:
        output = resolve(ROOT, args.json_out)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps({"errors":e,"warnings":w,"info":info},indent=2,ensure_ascii=False),encoding="utf-8")
    return 0 if not e else 2

if __name__ == "__main__":
    sys.exit(main())
