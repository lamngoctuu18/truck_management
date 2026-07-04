"""Module nhận diện biển số (6.4).

Chiến lược 2 tầng:
  1) Detector YOLO chuyên biển số (models/license_plate_detector.pt) cắt vùng biển số
     trong crop của xe -> tăng độ chính xác so với dò cả khung.
  2) OCR đọc ký tự. Ưu tiên dùng fast-alpr (ONNX, chạy CPU/GPU nhẹ). Nếu OCR của
     fast-alpr không có sẵn thì fallback sang chính detector+heuristic.

Kết quả trả về: (plate_text, confidence). Chuẩn hoá định dạng biển số VN.
"""
import re
import cv2
import numpy as np

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger("LPR")

# Ký tự hợp lệ cho biển số VN
_VALID = set("ABCDEFGHKLMNPRSTUVXYZ0123456789")
_PLATE_RE = re.compile(r"[^A-Z0-9]")
_VN_PLATE_RE = re.compile(r"^[1-9][0-9][A-Z]{1,2}[0-9]{5}$")


def normalize_plate(text: str) -> str:
    """Chuẩn hoá: viết hoa, bỏ ký tự lạ, sửa nhầm phổ biến của OCR."""
    if not text:
        return ""
    t = text.upper().strip()
    t = _PLATE_RE.sub("", t)
    # Chỉ giữ ký tự hợp lệ
    t = "".join(c for c in t if c in _VALID)
    return t


def is_valid_plate(text: str) -> bool:
    """Kiểm tra cấu trúc phổ biến của biển số Việt Nam sau chuẩn hoá.

    Chấp nhận dạng xe hiện đại 77C13558, 29LD12345...; loại biển thiếu số và
    các chuỗi OCR đảo vị trí
    chữ/số như 58071A4 thường sinh ra từ vùng không phải biển số.
    """
    return bool(_VN_PLATE_RE.fullmatch(text or ""))


def _safe_conf(value) -> float:
    """Chuẩn hoá confidence về float. fast-alpr đôi khi trả list (điểm từng ký tự)
    hoặc numpy array -> lấy trung bình. None/lỗi -> 0.0."""
    if value is None:
        return 0.0
    try:
        # list/tuple/ndarray -> trung bình các phần tử số
        if isinstance(value, (list, tuple)):
            nums = [float(v) for v in value if isinstance(v, (int, float))]
            return sum(nums) / len(nums) if nums else 0.0
        # numpy array / scalar
        if hasattr(value, "mean") and hasattr(value, "size"):
            return float(value.mean()) if value.size else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def select_best_plate(candidates):
    """Gộp kết quả OCR nhiều frame và chọn biển có đồng thuận tốt nhất.

    Mỗi candidate là dict ``{plate, confidence, plate_img}``. Kết quả giữ ảnh
    của lần đọc rõ nhất, đồng thời trả thêm ``votes`` để pipeline biết đã đủ
    số frame đồng thuận hay chưa.
    """
    groups = {}
    for item in candidates or []:
        plate = normalize_plate((item or {}).get("plate", ""))
        if not is_valid_plate(plate):
            continue
        conf = _safe_conf((item or {}).get("confidence", 0.0))
        group = groups.setdefault(plate, {"items": [], "score": 0.0})
        group["items"].append(item)
        group["score"] += conf
    if not groups:
        return None

    # Phiếu lặp lại được thưởng nhẹ; confidence vẫn quyết định khi số phiếu bằng nhau.
    plate, group = max(
        groups.items(),
        key=lambda kv: (len(kv[1]["items"]), kv[1]["score"],
                        max(_safe_conf(x.get("confidence", 0.0))
                            for x in kv[1]["items"])),
    )
    strongest = max(group["items"],
                    key=lambda x: _safe_conf(x.get("confidence", 0.0)))
    return {
        "plate": plate,
        "confidence": _safe_conf(strongest.get("confidence", 0.0)),
        "plate_img": strongest.get("plate_img"),
        "votes": len(group["items"]),
    }


class PlateRecognizer:
    def __init__(self):
        self._detector = None
        self._alpr = None
        self._ocr_backend = None
        self._init_backend()

    def _init_backend(self):
        # Ưu tiên fast-alpr: gói cả detector + OCR biển số
        try:
            from fast_alpr import ALPR
            self._alpr = ALPR(
                detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="global-plates-mobile-vit-v2-model",
            )
            self._ocr_backend = "fast_alpr"
            log.info("LPR backend: fast-alpr (ONNX)")
            return
        except Exception as e:  # noqa
            log.warning("fast-alpr khong san sang (%s), thu YOLO plate + OCR.", e)

        # Fallback: YOLO detector biển số riêng + OCR đơn giản (nếu có easyocr)
        try:
            from ultralytics import YOLO
            import os
            if os.path.exists(settings.PLATE_MODEL):
                self._detector = YOLO(settings.PLATE_MODEL)
                log.info("LPR: dung YOLO plate detector %s", settings.PLATE_MODEL)
        except Exception as e:  # noqa
            log.warning("Khong load duoc YOLO plate detector: %s", e)
        self._ocr_backend = "fallback"

    def read(self, vehicle_crop: np.ndarray):
        """Đọc biển số từ ảnh crop của một chiếc xe.

        Trả về dict: {plate, confidence, plate_img} hoặc None.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        if self._ocr_backend == "fast_alpr" and self._alpr is not None:
            return self._read_fast_alpr(vehicle_crop)
        return self._read_fallback(vehicle_crop)

    def _read_fast_alpr(self, crop):
        try:
            results = self._alpr.predict(crop)
        except Exception as e:  # noqa
            log.debug("fast-alpr predict loi: %s", e)
            return None
        if not results:
            return None
        best = None
        for r in results:
            ocr = getattr(r, "ocr", None)
            if ocr is None:
                continue
            text = normalize_plate(getattr(ocr, "text", "") or "")
            conf = _safe_conf(getattr(ocr, "confidence", 0.0))
            if not is_valid_plate(text) or conf < settings.OCR_MIN_CONF:
                continue
            if best is None or conf > best["confidence"]:
                # Cắt ảnh biển số nếu có bbox
                plate_img = None
                det = getattr(r, "detection", None)
                if det is not None and getattr(det, "bounding_box", None) is not None:
                    bb = det.bounding_box
                    try:
                        x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
                        plate_img = crop[max(0, y1):y2, max(0, x1):x2].copy()
                    except Exception:  # noqa
                        plate_img = None
                best = {"plate": text, "confidence": conf, "plate_img": plate_img}
        return best

    def _read_fallback(self, crop):
        """Fallback: nếu có YOLO plate detector thì cắt vùng biển; OCR bằng easyocr nếu có."""
        plate_region = crop
        conf_det = 0.5
        if self._detector is not None:
            try:
                res = self._detector.predict(crop, conf=settings.PLATE_CONF,
                                             verbose=False)[0]
                if len(res.boxes) > 0:
                    b = res.boxes[0]
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    conf_det = float(b.conf[0])
                    plate_region = crop[max(0, y1):y2, max(0, x1):x2]
            except Exception as e:  # noqa
                log.debug("plate detector loi: %s", e)

        text, ocr_conf = self._ocr_easyocr(plate_region)
        if not text:
            return None
        return {
            "plate": text,
            "confidence": min(conf_det, ocr_conf) if ocr_conf else conf_det * 0.6,
            "plate_img": plate_region.copy() if plate_region is not None else None,
        }

    _easyocr_reader = None

    def _ocr_easyocr(self, img):
        if img is None or img.size == 0:
            return "", 0.0
        try:
            if PlateRecognizer._easyocr_reader is None:
                import easyocr
                PlateRecognizer._easyocr_reader = easyocr.Reader(["en"], gpu=True)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            out = PlateRecognizer._easyocr_reader.readtext(gray, detail=1)
            texts = sorted(out, key=lambda x: -x[2])
            for _, t, c in texts:
                nt = normalize_plate(t)
                if is_valid_plate(nt):
                    return nt, float(c)
            if texts:
                return normalize_plate(texts[0][1]), float(texts[0][2])
        except Exception as e:  # noqa
            log.debug("easyocr loi/khong co: %s", e)
        return "", 0.0


# Singleton
_recognizer = None


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = PlateRecognizer()
    return _recognizer
