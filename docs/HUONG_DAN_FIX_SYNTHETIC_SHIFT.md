# V3 Spatial – Fix `synthetic_shift` broadcasting

## Ket qua bao cao da gui
- `evaluation_val.json`: `mode=strict`, `n=26`, Accuracy=1.0, Known=17/17, UNKNOWN=9/9. **VAL khong phai TEST.**
- `diagnose_v3_spatial.json`: ky hieu `xin_chao` cua S01 (nguoi TRAIN), 7/7 `wrong_position`, 2/2 `wrong_handshape`, 2/2 `wrong_orientation`, 3/3 `partial` cho `UNKNOWN`; 2/2 dung cho `xin_chao`. Day la diagnostic, **khong duoc tinh la ket qua unseen signer TEST**.
- `synthetic_shift`: `n=0` do NumPy broadcasting error, chua duoc danh gia.

## Nguyen nhan
Cu: `arr[exists,:21,0] += shift * widths[exists]`

Khi `arr[exists,:21,0]` co kich thuoc `(N_valid,21)`, `widths[exists]` co kich thuoc `(N_valid,)`. NumPy khong the broadcast hai kich thuoc nay, ke ca `(0,21)` va `(0,)`.

Moi: `arr[valid_hand,:,0] += (shift * widths[valid_hand])[:,None]`.

Ban va sua luon tinh huong tay vắng mat, Pose khong hop le. Tinh lai 453D sau khi dich toa do, chong lech giua Spatial60 va legacy feature.

## Cai dat
1. Backup file `scripts/diagnose_position_sensitivity.py` hien tai.
2. Giai nen ZIP vao root project, chi can chep de `scripts/diagnose_position_sensitivity.py`; chep them file test neu muon.
3. Kiem tra:

```powershell
python -m pytest -q tests/test_diagnose_shift_fix.py
```

4. Chay lai:

```powershell
python scripts/diagnose_position_sensitivity.py --min-shift-shoulders 1.0 --synthetic-shift 1.0
```

5. Mo `logs/diagnose_v3_spatial.json`. Kiem tra:
   - `by_condition.synthetic_shift.n >= 1`
   - `error_count == 0` (neu co loi thi xem `errors[]` va `records[].error`)
   - `records[condition=synthetic_shift]` co `prediction`, `p_true_sign`, `drop`, `shoulder_shift`.

**Khong can train lai.** Day la patch cong cu diagnostic, khong thay doi model/scaler/dataset.

### Luu y khoa hoc
Synthetic shift co the day ban tay ra ngoai khung hinh, do do chi la kiem tra do nhay phan thuc chinh (counterfactual). Khong duoc dung lam bang chung ket qua webcam that. De chot khoa luan, giu VAL de chon tham so, sau do danh gia TEST theo signer doc lap (S09/S10) va them hard negatives that, co nhan `condition`.
