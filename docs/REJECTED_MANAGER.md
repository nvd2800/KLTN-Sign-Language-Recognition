# Phase 03 — Rejected Sample Manager

Mục tiêu: không xóa vĩnh viễn REJECT ngay. Thay vào đó chuyển chúng sang
`dataset/rejected/` và lưu index để có thể restore.

## 1. Xem danh sách REJECT

```powershell
python scripts/manage_rejected_samples.py --list
```

Xem một sample:

```powershell
python scripts/manage_rejected_samples.py --list --sample-id xin_chao_S01_SES01_001
```

## 2. Dry-run trước khi move

```powershell
python scripts/manage_rejected_samples.py --move
```

Lệnh trên CHƯA thay đổi file.

## 3. Move thật

```powershell
python scripts/manage_rejected_samples.py --move --yes
```

REJECT được chuyển:

```text
dataset/raw/<sign>/<subject>/<session>/<sample>
             ↓
dataset/rejected/<sign>/<subject>/<session>/<sample>
```

Index:

```text
dataset/rejected/rejected_index.json
```

## 4. Move một sample cụ thể

Dry-run:

```powershell
python scripts/manage_rejected_samples.py --move --sample-id SAMPLE_ID
```

Move thật:

```powershell
python scripts/manage_rejected_samples.py --move --sample-id SAMPLE_ID --yes
```

## 5. Restore toàn bộ

Dry-run:

```powershell
python scripts/manage_rejected_samples.py --restore
```

Restore thật:

```powershell
python scripts/manage_rejected_samples.py --restore --yes
```

## 6. Restore một sample

```powershell
python scripts/manage_rejected_samples.py --restore --sample-id SAMPLE_ID --yes
```

## Cảnh báo

Sau khi move sample ra khỏi `dataset/raw`, nếu chạy lại Phase-2 audit gốc thì
raw dataset sẽ không còn đủ 168 sample. Đây là điều bình thường vì bạn đã
quarantine dữ liệu.

Nếu cần tái lập audit Phase 02 ban đầu:
1. restore sample,
2. chạy lại Phase 02 audit.

Không xóa `rejected_index.json`.
