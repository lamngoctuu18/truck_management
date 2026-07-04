"""Module nhận diện + tracking xe (6.1, 6.2).

Dùng YOLOv8 với chế độ track (ByteTrack tích hợp trong ultralytics) để vừa phát hiện
vừa gán track_id ổn định qua nhiều frame -> tránh đếm trùng.
"""
import numpy as np
from ultralytics import YOLO

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("Detector")

# COCO class id của phương tiện có thể chở vật liệu
# 2=car (đôi khi xe tải nhỏ bị nhận là car), 5=bus, 7=truck
COCO_VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

# Từ khoá tên lớp xe cần đếm (dùng cho model fine-tuned có lớp riêng)
VEHICLE_KEYWORDS = ("truck", "ben", "dump", "container", "trailer", "semi",
                    "lorry", "xe", "bus", "van", "loader", "excavator", "mixer",
                    "vehicle", "car")


class VehicleDetector:
    def __init__(self, model_path: str = None, device: str = None):
        model_path = model_path or settings.VEHICLE_MODEL
        self.model = YOLO(model_path)
        self.device = self._resolve_device(device or settings.DEVICE)
        self.target_ids = self._resolve_target_classes()
        log.info("VehicleDetector: model=%s device=%s classes=%s",
                 model_path, self.device,
                 {i: self.model.names[i] for i in self.target_ids})

    def _resolve_target_classes(self):
        """Tự xác định class id cần đếm.

        - Model COCO chuẩn (80 lớp, có 'truck' ở id 7): dùng {2,5,7}.
        - Model fine-tuned (lớp riêng): lấy mọi lớp có tên chứa từ khoá xe;
          nếu không match từ khoá nào thì lấy TẤT CẢ lớp (giả định model chỉ train xe).
        """
        names = self.model.names  # dict id->name
        is_coco = len(names) >= 80 and str(names.get(7, "")).lower() == "truck"
        if is_coco:
            configured = set(settings.COUNT_CLASSES)
            matched = [
                i for i, name in COCO_VEHICLE_CLASSES.items()
                if not configured or name.lower() in configured
            ]
            return sorted(matched or COCO_VEHICLE_CLASSES.keys())
        matched = [i for i, n in names.items()
                   if any(k in str(n).lower() for k in VEHICLE_KEYWORDS)]
        configured = set(settings.COUNT_CLASSES)
        configured_matches = [
            i for i in matched if str(names.get(i, "")).lower() in configured
        ]
        if configured_matches:
            matched = configured_matches
        if not matched:
            matched = list(names.keys())
        return sorted(matched)

    @staticmethod
    def _resolve_device(dev: str):
        if dev == "auto":
            try:
                import torch
                return 0 if torch.cuda.is_available() else "cpu"
            except Exception:  # noqa
                return "cpu"
        return dev

    def track(self, frame: np.ndarray):
        """Detect + track trên 1 frame. Trả về list dict:
        {track_id, cls_name, conf, bbox=(x1,y1,x2,y2), cx, cy}
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=self.target_ids,
            conf=settings.VEHICLE_CONF,
            device=self.device,
            verbose=False,
        )
        out = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return out
        boxes = r.boxes
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for tid, box, conf, cls in zip(ids, xyxy, confs, clss):
            x1, y1, x2, y2 = box.tolist()
            out.append({
                "track_id": int(tid),
                "cls_name": str(self.model.names.get(int(cls), "vehicle")),
                "conf": float(conf),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
            })
        return out

    def reset_tracking(self):
        """Reset trạng thái ByteTrack khi bắt đầu phiên hoặc đổi nguồn video.

        Ultralytics khởi tạo tracker lười trong predictor. Các phiên bản khác nhau
        có thể dùng list/tuple hoặc một tracker đơn, vì vậy thao tác theo hướng
        best-effort và không làm hỏng pipeline nếu API nội bộ thay đổi.
        """
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers is None:
            return
        if not isinstance(trackers, (list, tuple)):
            trackers = [trackers]
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
