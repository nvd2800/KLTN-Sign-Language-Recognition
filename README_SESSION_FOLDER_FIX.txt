FIX THEO DUNG THU MUC SESSION / SAMPLE ID (giong anh VS Code)
================================================================

YEU CAU:
  logs/feature_diagnostic/<label>/<subject>/<session>/<label>_<subject>_<session>_<NNN>/
  VD: logs/feature_diagnostic/cam_on/S07/SES01/cam_on_S07_SES01_006/

Trong moi sample:
  raw_landmarks.npz
  features.npz
  metadata.json  (condition, session, subject, sample_id, qc, video)
  source_video.avi (tu dong luu, neu encoder hoat dong)

Ben canh do:
  logs/feature_diagnostic/manifest.csv

Khong co --signer-alias. Cac phim 1..6 van giong cu.
Khong them condition lam thu muc; no nam trong metadata.json va manifest.csv.

1. Copy cac file core/scripts/configs trong ZIP vao goc project.
2. Chay:
  python scripts/live_feature_diagnostic.py --label cam_on --subject S07 --session SES01 --frames 32 --fullscreen
3. Review:
  python scripts/review_feature_diagnostic_videos.py --label cam_on --subject S07 --session SES01 --play
4. Xay lai manifest ca du lieu cu va moi:
  python scripts/rebuild_feature_diagnostic_manifest.py
5. Chuyen du lieu diagnostic cu sang thu muc Session (TUY CHON; sao luu truoc):
  python scripts/migrate_feature_diagnostic_layout.py
  python scripts/migrate_feature_diagnostic_layout.py --execute
  (neu metadata cu khong co session, co the them --default-session SES01,
   NHUNG chi khi chac chan nhung mau cu thuoc SES01)

Sau khi chuyen, rebuild manifest tu dong. Khong copy duplicate raw sang train.

Goi nay la PATCH, khong kem checkpoint model, dataset, hoac MediaPipe.
Xem docs/HUONG_DAN_CAU_TRUC_SESSION_SAMPLE_ID.md de biet them.
