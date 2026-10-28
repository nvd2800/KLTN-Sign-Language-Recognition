"""Convenience launcher for TCN V3.1; never silently switches to V3 baseline.

Examples:
    python scripts/live_phase09_v3_1.py --dry-run
    python scripts/live_phase09_v3_1.py --fullscreen
    python scripts/live_phase09_v3_1.py --fullscreen --target-label xin_chao --trial-isolation
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.check_phase09_v3_1 import preflight, resolve


def launch(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description="Start Phase09 realtime with TCN V3.1 (32x513) safely",add_help=True)
    parser.add_argument("--config",default="configs/phase9_v3_Final.yaml")
    parser.add_argument("--dataset",default="dataset/processed/temporal_v3_1_Final")
    parser.add_argument("--dry-run",action="store_true",help="Only verify artifacts; do not open camera")
    parser.add_argument("--check-model",action="store_true",help="Load keras model during pre-flight")
    args, forwarded = parser.parse_known_args(argv)

    cfg = resolve(root,args.config)
    errors, warnings, info = preflight(root,cfg, check_model=args.check_model, dataset=resolve(root,args.dataset))
    print("\n=== TCN V3.1 LIVE CHECK ===")
    for key in ("model","stats","labels","label_order","input_shape","checkpoint_dataset_counts","selected_dataset_counts"):
        if key in info: print(f"{key}: {info[key]}")
    for msg in warnings: print("WARNING:",msg)
    for msg in errors: print("ERROR:",msg)
    if errors: return 2
    print("PREFLIGHT: PASS")
    if args.dry_run: return 0
    live_script = root / 'scripts' / 'live_phase09_v3_spatial.py'
    if not live_script.is_file():
        print('ERROR: Phase09 live script missing:',live_script)
        return 2
    cmd = [sys.executable,str(live_script),'--config',str(cfg),*forwarded]
    print('STARTING PHASE09 WITH V3.1 CHECKPOINT...\n')
    return subprocess.call(cmd,cwd=str(root))


if __name__ == '__main__':
    sys.exit(launch())
