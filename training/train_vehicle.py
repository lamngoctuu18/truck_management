"""Fine-tune YOLO nhận diện xe cho dữ liệu thực tế nhà máy (mục 6.1).

Mục tiêu: dạy model phân biệt xe ben / xe tải / container... đặc thù nhà máy,
thay vì chỉ dùng nhãn COCO chung (car/truck/bus).

Chuẩn bị dữ liệu:
  1. Thu thập ảnh: bật COLLECT_TRAINING_DATA=true trong .env để hệ thống tự lưu
     ảnh crop xe vào data/training_raw/, HOẶC trích ảnh từ video bằng
     training/extract_frames.py.
  2. Gán nhãn bằng công cụ (khuyến nghị: LabelImg, Roboflow, CVAT) theo định dạng
     YOLO: mỗi ảnh 1 file .txt cùng tên, mỗi dòng "class cx cy w h" (toạ độ 0..1).
  3. Đặt ảnh + nhãn vào:
        data/datasets/vehicle/images/train, .../val
        data/datasets/vehicle/labels/train, .../val
  4. Sửa danh sách lớp trong data/datasets/vehicle/data.yaml (đã tạo sẵn mẫu).

Chạy:
  venv\\Scripts\\python.exe training\\train_vehicle.py --epochs 100 --model yolov8n.pt
"""
import os
import argparse

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_BASE, ".cache", "ultralytics"))

DATA_YAML = os.path.join(_BASE, "data", "datasets", "vehicle", "data.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="model gốc để fine-tune")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="giảm nếu thiếu VRAM (GTX 1650)")
    ap.add_argument("--device", default="0", help="0=GPU, cpu=CPU")
    args = ap.parse_args()

    if not os.path.exists(DATA_YAML):
        raise SystemExit(f"Chua co {DATA_YAML}. Xem training/README.md de chuan bi dataset.")

    from ultralytics import YOLO
    model = YOLO(os.path.join(_BASE, "models", args.model)
                 if os.path.exists(os.path.join(_BASE, "models", args.model))
                 else args.model)

    print(f"Fine-tune {args.model} tren {DATA_YAML} ...")
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=os.path.join(_BASE, "training", "runs"),
        name="vehicle_finetune",
        exist_ok=True,
    )
    # Sao chép model tốt nhất về thư mục models/
    best = os.path.join(_BASE, "training", "runs", "vehicle_finetune",
                        "weights", "best.pt")
    if os.path.exists(best):
        import shutil
        dst = os.path.join(_BASE, "models", "vehicle_finetuned.pt")
        shutil.copy(best, dst)
        print(f"\nDA LUU model tot nhat -> {dst}")
        print("Cap nhat .env: VEHICLE_MODEL=models/vehicle_finetuned.pt roi khoi dong lai.")


if __name__ == "__main__":
    main()
