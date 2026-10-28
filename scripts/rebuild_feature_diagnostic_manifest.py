"""Reindex old and new Feature Diagnostic samples without modifying raw files.

Usage: python scripts/rebuild_feature_diagnostic_manifest.py
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.feature_diagnostic_video import rebuild_trial_manifest

def main(argv=None):
    parser=argparse.ArgumentParser(description='Update manifest.csv for all existing diagnostic trials')
    parser.add_argument('--input',default='logs/feature_diagnostic')
    args=parser.parse_args(argv)
    root=Path(args.input)
    if not root.is_absolute(): root=ROOT/root
    if not root.is_dir():raise FileNotFoundError(root)
    result=rebuild_trial_manifest(root)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':main()
