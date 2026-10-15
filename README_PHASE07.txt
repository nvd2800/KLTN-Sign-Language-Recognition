PHASE 07 TEMPORAL MODEL — GRU / TCN

Install all package files into project root.

Run:
pip install -r requirements-phase7.txt

pytest -q tests/test_phase7_preprocessing.py
python scripts/prepare_phase7_dataset.py
python scripts/audit_phase7.py

python scripts/train_phase7.py --model gru
python scripts/train_phase7.py --model tcn

python scripts/evaluate_phase7.py --model gru
python scripts/evaluate_phase7.py --model tcn

python scripts/compare_phase7_models.py
python scripts/audit_phase7_models.py

Default:
sequence = 32 frames
feature = Hands+Pose+Face = 453/frame

GRU:
128 -> Dropout -> 64 -> Dense64 -> Softmax

TCN:
Dilated causal Conv -> residual -> GlobalMaxPool -> Dense -> Softmax
