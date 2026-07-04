"""Fine-tune YOLO nhận diện xe từ file ZIP dataset tải thủ công (không cần API key).

=== CÁCH DÙNG ===
1. Tải dataset xe ben/xe tải ở định dạng YOLOv8 (ZIP) từ:
   - Kaggle: https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow
     (bấm "Download", đăng nhập Kaggle miễn phí — KHÔNG cần API key Roboflow)
   - Hoặc Roboflow Universe: mở dataset -> Download Dataset -> chọn "YOLOv8" ->
     "Download zip to computer".
2. Bỏ file .zip vào thư mục:  training/dataset_zip/
3. Chạy:
   venv\\Scripts\\python.exe training\\train_from_zip.py --epochs 80 --batch 8

Script tự động:
  - Giải nén mọi .zip trong training/dataset_zip/
  - Tìm data.yaml, sửa đường dẫn train/val cho khớp máy bạn
  - Fine-tune YOLOv8, lưu model tốt nhất về models/vehicle_finetuned.pt
"""
import os
import glob
import zipfile
import argparse
import shutil

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_BASE, ".cache", "ultralytics"))

ZIP_DIR = os.path.join(_BASE, "training", "dataset_zip")
EXTRACT_DIR = os.path.join(_BASE, "data", "datasets", "truck_zip")


def _unzip_all():
    zips = glob.glob(os.path.join(ZIP_DIR, "*.zip"))
    if not zips:
        raise SystemExit(
            f"Khong tim thay file .zip nao trong {ZIP_DIR}\n"
            "Hay tai dataset YOLOv8 (ZIP) va bo vao thu muc do. Xem huong dan dau file.")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    for z in zips:
        print(f"Giai nen {os.path.basename(z)} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(EXTRACT_DIR)
    print(f"Da giai nen vao {EXTRACT_DIR}")


def _find_data_yaml():
    """Tìm data.yaml trong thư mục đã giải nén (kể cả thư mục con)."""
    candidates = glob.glob(os.path.join(EXTRACT_DIR, "**", "data.yaml"), recursive=True)
    candidates += glob.glob(os.path.join(EXTRACT_DIR, "**", "*.yaml"), recursive=True)
    # ưu tiên file tên data.yaml
    candidates = sorted(set(candidates),
                        key=lambda p: (os.path.basename(p) != "data.yaml", len(p)))
    return candidates[0] if candidates else None


def _fix_yaml_paths(yaml_path):
    """Đảm bảo train/val trong data.yaml trỏ đúng thư mục ảnh thực tế."""
    import yaml
    root = os.path.dirname(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def _resolve(split_key, fallbacks):
        val = cfg.get(split_key)
        # thử theo giá trị sẵn có
        cands = []
        if val:
            cands.append(os.path.join(root, str(val).replace("../", "")))
            cands.append(os.path.join(root, str(val)))
        for fb in fallbacks:
            cands.append(os.path.join(root, fb))
        for c in cands:
            if os.path.isdir(c):
                return c
        return None

    train_dir = _resolve("train", ["train/images", "images/train", "train"])
    val_dir = _resolve("val", ["valid/images", "val/images", "images/val",
                               "valid", "val"])
    if not train_dir:
        raise SystemExit(f"Khong tim thay thu muc anh train trong {root}. "
                         "Kiem tra lai cau truc dataset.")
    if not val_dir:
        # nếu không có val, dùng tạm train làm val (đủ để chạy)
        val_dir = train_dir
        print("CANH BAO: khong co tap val, dung tam tap train de val.")

    cfg["path"] = root
    cfg["train"] = os.path.relpath(train_dir, root).replace("\\", "/")
    cfg["val"] = os.path.relpath(val_dir, root).replace("\\", "/")

    fixed = os.path.join(root, "data_fixed.yaml")
    with open(fixed, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    names = cfg.get("names")
    print(f"Cac lop trong dataset: {names}")
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--base-model", default="yolov8n.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--skip-unzip", action="store_true",
                    help="bo qua giai nen (neu da giai nen truoc do)")
    args = ap.parse_args()

    if not args.skip_unzip:
        _unzip_all()

    yaml_path = _find_data_yaml()
    if not yaml_path:
        raise SystemExit(f"Khong tim thay data.yaml trong {EXTRACT_DIR}")
    print(f"Dung config: {yaml_path}")
    fixed_yaml = _fix_yaml_paths(yaml_path)

    from ultralytics import YOLO
    base = os.path.join(_BASE, "models", args.base_model)
    model = YOLO(base if os.path.exists(base) else args.base_model)
    print(f"Fine-tune {args.base_model} (epochs={args.epochs}, batch={args.batch}) ...")
    model.train(
        data=fixed_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=os.path.join(_BASE, "training", "runs"),
        name="truck_zip",
        exist_ok=True,
    )

    best = os.path.join(_BASE, "training", "runs", "truck_zip", "weights", "best.pt")
    if os.path.exists(best):
        dst = os.path.join(_BASE, "models", "vehicle_finetuned.pt")
        shutil.copy(best, dst)
        print("\n" + "=" * 60)
        print(f"XONG! Model da luu -> {dst}")
        print("Cap nhat .env:  VEHICLE_MODEL=models/vehicle_finetuned.pt")
        print("Roi khoi dong lai he thong (run.bat).")
        print("=" * 60)


if __name__ == "__main__":
    main()
