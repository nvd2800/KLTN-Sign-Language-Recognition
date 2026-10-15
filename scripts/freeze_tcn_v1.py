from __future__ import annotations
import argparse, hashlib, json, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tcn_v1_160samples")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = ROOT / "artifacts" / "tcn"
    dst = ROOT / "artifacts" / "experiments" / args.name
    temporal = ROOT / "dataset" / "processed" / "temporal_phase7"

    required = [
        src/"model.keras", src/"history.json", src/"training_config.json",
        src/"test_metrics.json", src/"confusion_matrix.csv",
        temporal/"preprocess_stats.npz", temporal/"labels.json", temporal/"dataset_summary.json",
        temporal/"test.npz",
    ]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    if dst.exists():
        if not args.force:
            raise FileExistsError(f"{dst} exists. Use --force only intentionally.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for p in required:
        shutil.copy2(p, dst/p.name)

    # Freeze exact unseen-signer TEST sample identities.
    import numpy as np
    with np.load(temporal/"test.npz", allow_pickle=False) as d:
        test_ids=[str(x) for x in d["sample_ids"].tolist()]
    (dst/"locked_test_sample_ids.json").write_text(
        json.dumps({"sample_ids":test_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    manifest={
        "experiment_name":args.name,
        "frozen_at":time.strftime("%Y-%m-%d %H:%M:%S"),
        "locked_test_set":True,
        "files":{},
    }
    for p in dst.iterdir():
        if p.is_file():
            manifest["files"][p.name]={"bytes":p.stat().st_size,"sha256":sha256(p)}
    (dst/"freeze_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    print("\nTCN V1 FREEZE")
    print("="*60)
    print("Destination :",dst)
    print("Status      : PASS")
    print("="*60)

if __name__=="__main__":
    main()
