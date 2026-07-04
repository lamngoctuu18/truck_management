"""Module phân loại trạng thái tải (6.5).

POC: heuristic dựa trên phân tích ảnh thùng xe (độ "đầy" / kết cấu bề mặt vùng thùng).
Đây là bản khung có thể thay bằng model phân loại chuyên biệt sau khi có dữ liệu gán nhãn.

Trả về: (cargo_type, load_status)
  cargo_type: SOIL | BRICK | EMPTY | UNKNOWN
  load_status: LOADED | EMPTY | UNKNOWN
"""
import cv2
import numpy as np
from app.core.logger import get_logger

log = get_logger("LoadClassifier")

# Ánh xạ nhãn model -> (cargo_type, load_status)
_CLS_MAP = {
    "empty": ("EMPTY", "EMPTY"),
    "soil": ("SOIL", "LOADED"),
    "brick": ("BRICK", "LOADED"),
    "covered": ("UNKNOWN", "LOADED"),
    "other": ("UNKNOWN", "UNKNOWN"),
}


class LoadClassifier:
    def __init__(self):
        self._model = None
        self._try_load_model()

    def _try_load_model(self):
        """Nếu có model đã train (models/load_classifier.pt) thì dùng, không thì heuristic."""
        import os
        from app.core.config import BASE_DIR
        path = os.path.join(str(BASE_DIR), "models", "load_classifier.pt")
        if os.path.exists(path):
            try:
                from ultralytics import YOLO
                self._model = YOLO(path)
                log.info("LoadClassifier: dung model da train %s", path)
            except Exception as e:  # noqa
                log.warning("Khong load duoc load_classifier.pt: %s -> dung heuristic", e)

    def classify(self, vehicle_crop: np.ndarray, cls_name: str = "truck"):
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "UNKNOWN", "UNKNOWN"

        # Ưu tiên model đã train
        if self._model is not None:
            try:
                r = self._model.predict(vehicle_crop, verbose=False)[0]
                if r.probs is not None:
                    label = r.names[int(r.probs.top1)].lower()
                    return _CLS_MAP.get(label, ("UNKNOWN", "UNKNOWN"))
            except Exception as e:  # noqa
                log.debug("load model predict loi: %s", e)

        return self._classify_heuristic(vehicle_crop)

    def _classify_heuristic(self, vehicle_crop):

        # Lấy phần thùng xe: nửa trên của crop (với xe ben nhìn nghiêng/sau)
        h, w = vehicle_crop.shape[:2]
        cargo_region = vehicle_crop[: max(1, h // 2), :]

        try:
            gray = cv2.cvtColor(cargo_region, cv2.COLOR_BGR2GRAY)
        except Exception:  # noqa
            return "UNKNOWN", "UNKNOWN"

        # Độ nhám bề mặt (biến thiên Laplacian): đất/gạch có kết cấu -> variance cao,
        # thùng rỗng (kim loại phẳng) -> variance thấp.
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_intensity = float(gray.mean())

        # Ngưỡng heuristic (cần tinh chỉnh bằng dữ liệu thực tế tại nhà máy)
        if lap_var < 60:
            return "EMPTY", "EMPTY"

        # Có tải: phân biệt sơ bộ đất (tối, nâu) vs gạch (đỏ/cam sáng hơn)
        b, g, r = cv2.split(cargo_region.astype(np.int32))
        redness = float((r - (g + b) / 2).mean())
        if redness > 15 and mean_intensity > 90:
            return "BRICK", "LOADED"
        return "SOIL", "LOADED"


_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = LoadClassifier()
    return _classifier
