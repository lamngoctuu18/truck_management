"""Train model phân loại trạng thái tải (mục 6.5).

Thay thế heuristic hiện tại bằng model phân loại ảnh (YOLOv8-cls) để phân biệt:
  empty (rỗng) / soil (có đất) / brick (có gạch) / covered (phủ bạt) / other.

Chuẩn bị dữ liệu (phân loại ảnh theo thư mục):
  data/datasets/load_cls/
      train/
          empty/    *.jpg
          soil/     *.jpg
          brick/    *.jpg
          covered/  *.jpg
      val/
          empty/ ... (tương tự)

  Nguồn ảnh: bật COLLECT_TRAINING_DATA=true để hệ thống tự lưu crop xe khi qua vạch,
  sau đó phân loại ảnh vào đúng thư mục theo mắt thường.

Chạy:
  venv\\Scripts\\python.exe training\\train_load_classifier.py --epochs 50
"""
import os
import argparse

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_BASE, ".cache", "ultralytics"))

DATASET = os.path.join(_BASE, "data", "datasets", "load_cls")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-cls.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    train_dir = os.path.join(DATASET, "train")
    if not os.path.isdir(train_dir) or not os.listdir(train_dir):
        raise SystemExit(
            f"Chua co du lieu train tai {train_dir}.\n"
            "Tao cac thu muc con: empty/ soil/ brick/ covered/ va bo anh vao.")

    from ultralytics import YOLO
    model = YOLO(args.model)   # yolov8n-cls tự tải
    print(f"Train phan loai tai tren {DATASET} ...")
    model.train(
        data=DATASET,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=os.path.join(_BASE, "training", "runs"),
        name="load_cls",
        exist_ok=True,
    )
    best = os.path.join(_BASE, "training", "runs", "load_cls", "weights", "best.pt")
    if os.path.exists(best):
        import shutil
        dst = os.path.join(_BASE, "models", "load_classifier.pt")
        shutil.copy(best, dst)
        print(f"\nDA LUU -> {dst}")
        print("He thong se tu dong dung model nay neu ton tai (xem load_classifier.py).")


if __name__ == "__main__":
    main()
