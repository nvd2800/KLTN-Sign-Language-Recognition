from pathlib import Path
import yaml
import tensorflow as tf

# Đọc đường dẫn từ file cấu hình của bạn
ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "configs/phase8_tcn_v2.yaml").read_text(encoding="utf-8"))
model_path = ROOT / cfg["paths"]["experiment_dir"] / "last_model.keras"

print(f"Đang tải mô hình từ: {model_path}")
model = tf.keras.models.load_model(model_path)

# In ra cấu trúc các tầng (layers) và tổng số tham số
model.summary()