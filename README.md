# Hệ thống AI bóc tách thông tin từ video — Đếm xe chở đất/gạch vào–ra nhà máy

Bản POC theo kế hoạch triển khai: nhận diện xe (YOLOv8) → tracking (ByteTrack) →
đếm line-crossing IN/OUT → đọc biển số (fast-alpr) → phân loại tải → lưu bằng chứng
(ảnh/clip) → PostgreSQL → dashboard web realtime.

> ⚙️ Toàn bộ thư viện + model AI được cài trong `venv` và thư mục `.cache` **trên ổ E**
> (không đụng ổ C đang đầy). GPU: chạy CUDA nếu có (đã test trên GTX 1650).

---

## 1. Yêu cầu
- Python 3.12 (cài từ python.org, tick **"Add Python to PATH"**)
- PostgreSQL đang chạy ở `localhost:5432`, user `postgres`, mật khẩu `123` (DB `vehicle_management` tự tạo)
- (Tùy chọn) GPU NVIDIA + CUDA để tăng tốc

## 2. Cài đặt (lần đầu)

Nhấp đúp file **`install.bat`** — script tự động:
- Tạo môi trường ảo `venv` trong thư mục project
- Cài PyTorch CUDA + ultralytics + fast-alpr + toàn bộ thư viện
- Hướng cache/model AI về ổ E (tránh ổ C đầy)

> Lần cài đầu tải ~2.5GB (PyTorch), mất vài phút. Chỉ cần chạy **1 lần**.

## 3. Chạy hệ thống

Nhấp đúp **`run.bat`**, hoặc:

```bash
venv\Scripts\python.exe run.py
```

Mở trình duyệt: **http://localhost:8000**

### Chế độ demo đếm xe vào/ra

Dashboard chính dùng **bộ đếm theo phiên**: đổi nguồn hoặc bấm `Reset bộ đếm` sẽ
đưa IN/OUT về 0 nhưng không xoá lịch sử trong PostgreSQL. Với file video, hệ thống dừng
ở frame cuối; bấm `Phát lại video` để bắt đầu một phiên mới.

Để ưu tiên FPS và độ ổn định khi chỉ demo đếm xe, cấu hình mặc định tắt OCR biển số,
phân loại tải và ghi clip. Có thể bật lại từng chức năng trong `.env`.

## 3. Cấu hình — file `.env`

| Biến | Ý nghĩa |
|------|---------|
| `VIDEO_SOURCE` | Nguồn video: `0` (webcam), đường dẫn file (`data/videos/x.mp4`), hoặc URL `rtsp://...` |
| `CAMERA_TYPE` | `GATE` / `GATE_IN` / `GATE_OUT` / `WEIGHBRIDGE` / `YARD` / `QUEUE` |
| `FPS_PROCESS` | Số khung hình/giây đưa vào AI (giảm để đỡ tải GPU) |
| `DEDUP_SECONDS` | Khoảng chống đếm trùng (giây) |
| `DEMO_MODE` | Bật giao diện/luồng chạy ưu tiên cho demo đếm xe |
| `VIDEO_LOOP` | `false`: dừng khi hết file; `true`: phát lặp và reset phiên |
| `PLAYBACK_REALTIME` | Phát file theo FPS gốc để demo không chạy quá nhanh |
| `COUNT_CLASSES` | Danh sách lớp cần đếm, ví dụ `car,bus,truck` |
| `ENABLE_PLATE_RECOGNITION` | Bật/tắt OCR biển số; nguồn camera thật mặc định bật |
| `LPR_MAX_ATTEMPTS` | Số lần OCR tối đa cho mỗi track |
| `LPR_INTERVAL_SEC` | Khoảng nghỉ giữa hai lần OCR cùng một track |
| `LPR_MIN_VOTES` | Số frame đồng thuận để chốt biển số |
| `ENABLE_LOAD_CLASSIFICATION` | Bật/tắt phân loại đất/gạch/rỗng |
| `ENABLE_EVIDENCE_CLIPS` | Bật/tắt ghi clip; snapshot vẫn luôn được lưu |
| `ENABLE_ANALYTICS_ALERTS` | Bật/tắt cảnh báo nghiệp vụ trong phiên demo |
| `LINE_HYSTERESIS_PX` | Vùng chết quanh vạch để chống đếm do bbox rung |
| `IN_DIRECTION` | `down` = xe đi trên→dưới là VÀO; `up` = ngược lại |
| `DB_*` | Kết nối PostgreSQL |

Sau khi sửa `.env`, khởi động lại hệ thống.

## 4. Các màn hình dashboard
- **Tổng quan** (`/`): camera trực tiếp có vẽ vạch ảo + bbox, số xe vào/ra/còn bãi, feed sự kiện realtime.
- **Sự kiện xe** (`/events`): danh sách lượt xe, ảnh/clip bằng chứng, **hiệu chỉnh biển số** khi AI đọc sai.
  Có thể bấm **Đọc** để chạy lại OCR cho một snapshot hoặc **Đọc lại 50 biển thiếu**
  để bổ sung dữ liệu cũ. Trạng thái phân biệt rõ OCR tắt, không thấy biển và đọc thành công.
- **Chuyến xe** (`/trips`): ghép lượt vào + ra theo biển số → 1 chuyến, tính thời gian lưu bãi.
- **Xe chờ lâu** (`/waiting`): theo dõi realtime xe đứng yên trong vùng chờ, thời gian chờ, cảnh báo khi vượt ngưỡng (6.6).
- **Đối soát cân** (`/reconcile`): nạp phiếu cân từ CSV/Excel hoặc nhập tay, tự đối soát biển số + thời gian với sự kiện xe AI (Giai đoạn 6).
- **Cảnh báo** (`/alerts`): 10 loại — xe ra không có lượt vào, **xe vào không ra sau X giờ**, **xe quay đầu**, **biển số ở 2 nơi**, **lệch phiếu cân**, xe chờ lâu, biển số confidence thấp, camera mất tín hiệu…
- **Báo cáo** (`/reports`): tổng hợp theo ngày, xuất CSV.
- **Cấu hình** (`/config`): **kéo chuột để đặt lại vạch ảo** trực tiếp trên khung hình.

## 5. Cấu trúc mã nguồn

```
app/
  core/       config, logger, pipeline (vòng lặp xử lý video)
  models_ai/  vehicle_detector, line_counter, plate_recognizer,
              load_classifier, evidence (snapshot/clip)
  db/         models (5 bảng theo mục 7 kế hoạch), database
  api/        event_bus (cầu nối realtime), main (FastAPI + routes)
web/
  templates/  6 trang HTML + _sidebar
  static/     style.css, app.js
models/       yolov8n.pt, (license_plate_detector.pt nếu có)
data/         videos, snapshots, clips
.cache/       cache model AI (đặt trên ổ E)
```

## 6. Model AI sử dụng
- **Nhận diện xe**: YOLOv8n (`models/yolov8n.pt`, tự tải) — lớp car/bus/truck.
- **Tracking**: ByteTrack (tích hợp trong Ultralytics).
- **Đọc biển số**: `fast-alpr` — detector biển số YOLOv9 + OCR MobileViT (ONNX).
  Fallback: YOLO plate detector + EasyOCR nếu cần.
- **Phân loại tải**: heuristic phân tích thùng xe (khung có sẵn để thay bằng model
  chuyên biệt sau khi có dữ liệu gán nhãn tại nhà máy).
- **Phát hiện xe chờ lâu (6.6)**: thuật toán theo dõi xe đứng yên trong vùng chờ
  (không cần model). Cấu hình ROI + ngưỡng trong bảng `zone_config` / API `/api/config/zone`.

## 7. Tích hợp cân xe / phiếu xuất kho (Giai đoạn 6)
Vào trang **Đối soát cân** (`/reconcile`):
- **Nạp file**: kéo file CSV/Excel do phần mềm cân xuất. Hệ thống tự nhận diện cột
  (hỗ trợ tên tiếng Việt: "Bien so", "So phieu", "Trong luong vao"…) và đối soát ngay.
- **Nhập tay**: điền phiếu trực tiếp trên web.
- **Đối soát**: ghép phiếu với sự kiện xe theo biển số (đã chuẩn hoá) + thời gian trong
  cửa sổ `RECONCILE_TIME_WINDOW_MIN` phút. Trạng thái: KHỚP / LỆCH / CHƯA GHÉP.
  Phiếu lệch biển số sinh cảnh báo `RECONCILE_MISMATCH`.

## 8. Huấn luyện model với dữ liệu nhà máy
Xem [training/README.md](training/README.md). Tóm tắt:
- Đặt `COLLECT_TRAINING_DATA=true` trong `.env` để tự lưu ảnh crop xe khi chạy.
- `training/train_vehicle.py` — fine-tune nhận diện xe (xe ben, container…).
- `training/train_load_classifier.py` — train phân loại tải (đất/gạch/rỗng).
- Model train xong tự về `models/`, hệ thống tự dùng.

## 9. Kiểm thử nhanh (không cần web)
```bash
venv\Scripts\python.exe test_pipeline.py data\videos\test_traffic.mp4 150
```
In số xe detect, số lượt cross vạch, số sự kiện ghi DB.

## 8. Ghi chú độ chính xác
- Độ chính xác đọc biển số phụ thuộc **góc camera + ánh sáng + biển VN**. Với video giao
  thông chung/biển nước ngoài, OCR có thể không ra kết quả — đây là đúng thiết kế.
- Khi lắp camera thật đúng góc (thẳng đầu/đuôi xe, đủ sáng) và biển số VN, độ chính xác
  sẽ tăng đáng kể. Có thể fine-tune model cho biển VN ở giai đoạn sau.
