# Huấn luyện model với dữ liệu thực tế nhà máy

Thư mục này chứa script để fine-tune model cho đúng đặc thù nhà máy (Giai đoạn 3–4
của kế hoạch). Chỉ cần chạy khi bạn đã có **dữ liệu ảnh thật đã gán nhãn**.

## Bước 1: Thu thập ảnh (2 cách)

**Cách A — tự động khi hệ thống chạy** (khuyến nghị):
Trong `.env` đặt `COLLECT_TRAINING_DATA=true` rồi chạy hệ thống bình thường. Mỗi xe qua
vạch sẽ được lưu ảnh crop vào `data/training_raw/YYYYMMDD/`. Tên file có sẵn gợi ý loại
xe + trạng thái tải để bạn phân loại nhanh.

**Cách B — trích từ video:**
```bat
venv\Scripts\python.exe training\extract_frames.py data\videos\camera.mp4 --every 15
```

## Bước 2: Gán nhãn

### Cho model nhận diện xe (train_vehicle.py)
Dùng LabelImg / Roboflow / CVAT, xuất **định dạng YOLO**. Mỗi ảnh có 1 file `.txt` cùng
tên, mỗi dòng: `class_id cx cy w h` (toạ độ 0..1). Sửa danh sách lớp trong
`data/datasets/vehicle/data.yaml` (mẫu có sẵn: truck, dump_truck, container, trailer).

Bố trí:
```
data/datasets/vehicle/images/train/*.jpg   labels/train/*.txt
data/datasets/vehicle/images/val/*.jpg     labels/val/*.txt
```

### Cho model phân loại tải (train_load_classifier.py)
Chỉ cần **phân ảnh vào thư mục theo lớp** (không cần gán bbox):
```
data/datasets/load_cls/train/{empty,soil,brick,covered}/*.jpg
data/datasets/load_cls/val/{empty,soil,brick,covered}/*.jpg
```

## Cách nhanh: dùng dataset xe ben có sẵn (KHÔNG cần API key)

Nếu bạn muốn model phân biệt **xe ben / xe tải / máy công trình** mà chưa có dữ liệu
riêng, dùng dataset có sẵn:

1. Tải dataset YOLOv8 (ZIP) — không cần API key Roboflow:
   - **Kaggle** (đăng nhập miễn phí, bấm Download):
     https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow
   - Hoặc **Roboflow Universe**: mở dataset → Download Dataset → chọn **YOLOv8** →
     **"Download zip to computer"** (không cần code/API key).
2. Bỏ file `.zip` vào `training/dataset_zip/`
3. Chạy:
   ```bat
   venv\Scripts\python.exe training\train_from_zip.py --epochs 80 --batch 8
   ```
   Script tự giải nén, tìm `data.yaml`, sửa đường dẫn, train, và lưu model về
   `models/vehicle_finetuned.pt`.
4. Sửa `.env`: `VEHICLE_MODEL=models/vehicle_finetuned.pt` rồi khởi động lại.

> Hệ thống **tự động nhận diện** model mới có lớp riêng (dump_truck, truck…) và đếm
> đúng — không cần sửa code. Có API key Roboflow thì dùng `download_and_train_truck.py`
> để tải + train hoàn toàn tự động.

## Bước 3: Train (dữ liệu tự gán nhãn)

```bat
REM Fine-tune nhận diện xe (giảm --batch nếu GTX 1650 thiếu VRAM)
venv\Scripts\python.exe training\train_vehicle.py --epochs 100 --batch 8

REM Train phân loại tải
venv\Scripts\python.exe training\train_load_classifier.py --epochs 50
```

Model tốt nhất tự sao chép về `models/`:
- `models/vehicle_finetuned.pt` → sửa `.env`: `VEHICLE_MODEL=models/vehicle_finetuned.pt`
- `models/load_classifier.pt` → hệ thống **tự động dùng** nếu file tồn tại.

Khởi động lại hệ thống để áp dụng model mới.

## Lưu ý
- Toàn bộ output train (`training/runs/`) và dữ liệu (`data/datasets/`, `data/training_raw/`)
  nằm trên ổ E, không đụng ổ C.
- Cần tối thiểu ~vài trăm ảnh mỗi lớp để model ổn định. Càng nhiều dữ liệu thực tế,
  độ chính xác càng cao (đúng khuyến nghị mục 6.1, 6.5, 15 của kế hoạch).
