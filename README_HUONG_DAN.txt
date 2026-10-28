V3.1 HARD NEGATIVE CHECK - POWERSHELL FIX

1. Sao chep scripts/check_v3_hardnegative_split.py vao thu muc scripts/ cua project.
2. Mo PowerShell tren VS Code TAI THU MUC GOC project.
3. Chay 1 lenh (KHONG dan @' ... '@):
    python scripts/check_v3_hardnegative_split.py
4. Xem cac muc:
    TRAIN total valid hard negatives
    VAL total valid hard negatives
    TEST total valid hard negatives
    INVALID (metadata thieu nhan/trang thai, file NPZ thieu, sai folder/session)
5. Mo bao cao:
    logs/v3_hardnegative_audit/split_check.json

Luu y:
- Cong cu chi doc, khong sua/chuyen du lieu.
- Chinh xac nhu prepare: condition, subject, label phai co trong metadata.json.
- S08 VAL, S09-S10 TEST (neu mapping trong dataset_summary.json dung nhu vay).
- strict can TRAIN/VAL/TEST deu co hard negatives that.
- Neu VAL/TEST=0 thi bo sung mau that tu nguoi doc lap; khong doi subject/test thanh train.
- Neu dem split du nhung prepare van loi, can kiem tra shape/key trong npz theo source.
- Khong duoc dung --mode dev de bao cao ket qua chinh thuc.
