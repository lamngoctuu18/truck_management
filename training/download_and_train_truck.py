"""Tải dataset xe ben/xe tải có sẵn từ Roboflow Universe rồi fine-tune YOLO.

Model đầu ra phân biệt: dump_truck (xe ben), truck (xe tải), semi, wheel_loader,
excavator... tuỳ dataset — thay vì chỉ "truck" chung của COCO.

=== CÁCH DÙNG ===
1. Đăng ký tài khoản miễn phí tại https://roboflow.com và lấy API key:
   https://app.roboflow.com/settings/api  (Private API Key)
2. Chạy:
   venv\\Scripts\\python.exe training\\download_and_train_truck.py --api-key XXXX --epochs 80
   (hoặc đặt biến môi trường ROBOFLOW_API_KEY thay cho --api-key)

Script sẽ:
  - Tải dataset về data/datasets/truck_roboflow (trên ổ E)
  - Fine-tune YOLOv8 và lưu model tốt nhất về models/vehicle_finetuned.pt
  - In hướng dẫn cập nhật .env

=== DATASET MẶC ĐỊNH ===
"Construction Site Safety" (roboflow-universe-projects) — có các lớp xe công trình
gồm dump truck, excavator, wheel loader, truck... Bạn có thể đổi sang dataset khác
bằng --workspace/--project/--version (xem phần "DATASET KHÁC" ở cuối file).
"""
import os
import argparse
import shutil

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_BASE, ".cache", "ultralytics"))

# Dataset mặc định (có lớp dump truck / construction vehicles)
DEFAULT_WS = "roboflow-universe-projects"
DEFAULT_PROJECT = "construction-site-safety"
DEFAULT_VERSION = 27

DATASET_DIR = os.path.join(_BASE, "data", "datasets", "truck_roboflow")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY", ""),
                    help="Roboflow Private API Key (hoặc đặt ROBOFLOW_API_KEY)")
    ap.add_argument("--workspace", default=DEFAULT_WS)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--version", type=int, default=DEFAULT_VERSION)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8, help="giảm nếu GTX 1650 thiếu VRAM")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--base-model", default="yolov8n.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--download-only", action="store_true",
                    help="chỉ tải dataset, không train")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit(
            "Thieu API key. Lay tai https://app.roboflow.com/settings/api\n"
            "Roi chay:  ... --api-key <KEY>   hoac  set ROBOFLOW_API_KEY=<KEY>")

    # 1) Tải dataset
    print(f"Tai dataset {args.workspace}/{args.project} v{args.version} ...")
    from roboflow import Roboflow
    os.makedirs(DATASET_DIR, exist_ok=True)
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov8", location=DATASET_DIR)
    data_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"Da tai xong -> {dataset.location}")
    print(f"data.yaml: {data_yaml}")

    if args.download_only:
        print("Da tai xong (download-only). Xem cac lop trong data.yaml.")
        return

    # 2) Fine-tune
    from ultralytics import YOLO
    base = os.path.join(_BASE, "models", args.base_model)
    model = YOLO(base if os.path.exists(base) else args.base_model)
    print(f"Fine-tune {args.base_model} ...")
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=os.path.join(_BASE, "training", "runs"),
        name="truck_roboflow",
        exist_ok=True,
    )

    # 3) Lưu model tốt nhất
    best = os.path.join(_BASE, "training", "runs", "truck_roboflow",
                        "weights", "best.pt")
    if os.path.exists(best):
        dst = os.path.join(_BASE, "models", "vehicle_finetuned.pt")
        shutil.copy(best, dst)
        print("\n" + "=" * 60)
        print(f"XONG! Model da luu -> {dst}")
        print("Cap nhat .env:  VEHICLE_MODEL=models/vehicle_finetuned.pt")
        print("Roi khoi dong lai he thong.")
        print("=" * 60)


# === DATASET KHÁC (thay bằng --workspace/--project/--version) ===
# Tìm thêm tại: https://universe.roboflow.com/search?q=class:dump+truck
# Mỗi trang dataset có nút "Download this Dataset" -> hiện workspace/project/version.

if __name__ == "__main__":
    main()
