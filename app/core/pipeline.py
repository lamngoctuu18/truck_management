"""Pipeline AI Video Analytics (mục 4.2 - đếm theo sự kiện nghiệp vụ).

Luồng xử lý mỗi frame:
  1. Detect + track xe (ByteTrack)
  2. Vẽ overlay (bbox, vạch ảo, số đếm) -> dùng cho MJPEG stream
  3. Line-crossing -> phát sinh sự kiện IN/OUT
  4. Với mỗi sự kiện: đọc biển số, phân loại tải, lưu snapshot/clip, ghi DB
  5. Đẩy sự kiện qua hàng đợi realtime (WebSocket) + cập nhật bộ đếm chung

Chạy trong 1 luồng nền riêng, không chặn web server.
"""
import time
import threading
import uuid
from datetime import datetime, timedelta

import cv2

from app.core.config import settings
from app.core.logger import get_logger
from app.models_ai.vehicle_detector import VehicleDetector
from app.models_ai.line_counter import LineCounter
from app.models_ai.dwell_monitor import DwellMonitor
from app.models_ai.plate_recognizer import get_recognizer, select_best_plate
from app.models_ai.load_classifier import get_classifier
from app.models_ai.evidence import save_snapshot, ClipBuffer
from app.db.database import get_session
from app.db.models import VehicleEvent, VehicleTrip, AlertEvent, CameraConfig, LineConfig

log = get_logger("Pipeline")


def _open_source(src):
    """Mở nguồn video: webcam (số), file, hoặc RTSP."""
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    return cap


class AnalyticsPipeline:
    def __init__(self, event_bus=None):
        self.event_bus = event_bus          # để đẩy sự kiện realtime
        self.detector = None
        self.recognizer = None
        self.classifier = None
        self.counter = None
        self.dwell = None                   # giám sát xe chờ lâu (6.6)
        self.clip_buffer = None
        self.camera_id = 1
        self._roi_ratio = (0.1, 0.5, 0.9, 0.95)

        self._running = False
        self._thread = None
        self._latest_frame = None           # frame overlay mới nhất (cho MJPEG)
        self._frame_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.stats = {"in": 0, "out": 0, "in_yard": 0, "waiting": 0,
                      "alerts": 0, "fps": 0.0}
        self.recent_events = []             # cache sự kiện gần đây
        self.waiting_vehicles = []          # xe đang chờ (cho dashboard)
        self.anomaly = None                 # background checker (6.7)
        # OCR tích luỹ theo track: candidates/attempts/votes + kết quả tốt nhất.
        self._track_plates = {}
        self.video_source = settings.VIDEO_SOURCE   # nguồn video hiện tại (đổi runtime được)
        self._reload_source = False         # cờ yêu cầu đổi nguồn video
        self._reset_requested = False
        self._reset_reason = ""
        self.session_id = self._new_session_id()
        self.session_started_at = datetime.now()
        self.source_state = "idle"          # idle/starting/running/ended/offline/switching
        self.video_finished = False
        self.last_error = ""
        self.frame_width = 1280
        self.frame_height = 720
        self.source_fps = 0.0

    # ---------- vòng đời ----------
    def start(self):
        if self._running:
            return
        self._running = True
        self.source_state = "starting"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Khởi động bộ kiểm tra bất thường định kỳ (xe kẹt bãi...)
        if settings.ENABLE_ANALYTICS_ALERTS:
            from app.core.anomaly import AnomalyChecker
            self.anomaly = AnomalyChecker(self._raise_alert)
            self.anomaly.start()
        log.info("Pipeline started")

    def stop(self):
        self._running = False
        self.source_state = "stopped"
        if self.anomaly:
            self.anomaly.stop()
        if self._thread:
            self._thread.join(timeout=3)
        log.info("Pipeline stopped")

    def is_running(self):
        return self._running

    def switch_source(self, new_source):
        """Đổi nguồn video khi đang chạy (không cần restart server).
        new_source: đường dẫn file, số webcam (chuỗi/số), hoặc URL rtsp."""
        self.video_source = str(new_source)
        self.source_state = "switching"
        self.video_finished = False
        self._reload_source = True
        # nếu pipeline chưa chạy thì khởi động
        if not self._running:
            self.start()
        log.info("Yeu cau doi nguon video -> %s", new_source)
        return True

    @staticmethod
    def _new_session_id():
        return uuid.uuid4().hex[:10]

    def request_reset(self, reason="manual"):
        """Yêu cầu reset tại đầu vòng lặp pipeline để tránh đua trạng thái."""
        self._reset_reason = reason
        self._reset_requested = True

    def replay_source(self):
        """Mở lại nguồn hiện tại từ đầu và bắt đầu một phiên mới."""
        self.source_state = "switching"
        self.video_finished = False
        self._reload_source = True
        if not self._running:
            self.start()

    def session_status(self):
        with self._state_lock:
            return {
                "session_id": self.session_id,
                "started_at": self.session_started_at.strftime("%Y-%m-%d %H:%M:%S"),
                "source": str(self.video_source),
                "source_state": self.source_state,
                "video_finished": self.video_finished,
                "video_loop": settings.VIDEO_LOOP,
                "ocr_enabled": settings.ENABLE_PLATE_RECOGNITION,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "source_fps": round(self.source_fps, 2),
                "last_error": self.last_error,
                "stats": dict(self.stats),
            }

    def _reset_session_state(self, reason="manual"):
        """Reset số đếm và toàn bộ trạng thái tạm, không xoá lịch sử trong DB."""
        if self.counter is not None:
            self.counter.reset()
        if self.dwell is not None:
            self.dwell.reset()
        if self.detector is not None:
            try:
                self.detector.reset_tracking()
            except Exception as e:  # noqa
                log.debug("reset tracker loi: %s", e)
        self._track_plates = {}   # xoá biển tích luỹ của phiên cũ
        if settings.ENABLE_EVIDENCE_CLIPS:
            self.clip_buffer = ClipBuffer(fps=int(settings.FPS_PROCESS))
        else:
            self.clip_buffer = None
        with self._state_lock:
            self.stats.update({"in": 0, "out": 0, "in_yard": 0,
                               "waiting": 0, "alerts": 0, "fps": 0.0})
            self.waiting_vehicles = []
            self.recent_events = []
            self.session_id = self._new_session_id()
            self.session_started_at = datetime.now()
        self._reset_requested = False
        self._reset_reason = ""
        self._publish({"type": "session_reset", "reason": reason,
                       "session": self.session_status()})
        log.info("Bat dau phien dem moi %s (%s)", self.session_id, reason)

    def _set_source_state(self, state, error=""):
        changed = state != self.source_state or error != self.last_error
        self.source_state = state
        self.video_finished = state == "ended"
        self.last_error = error
        if changed:
            self._publish({"type": "source_status", "session": self.session_status()})

    def _is_file_source(self):
        src = str(self.video_source).lower()
        return (not src.isdigit() and
                src.endswith((".mp4", ".avi", ".mov", ".mkv")))

    # ---------- khởi tạo model ----------
    def _lazy_init(self):
        self.detector = VehicleDetector()
        self.recognizer = (get_recognizer()
                           if settings.ENABLE_PLATE_RECOGNITION else None)
        self.classifier = (get_classifier()
                           if settings.ENABLE_LOAD_CLASSIFICATION else None)
        self._load_line_config()
        self._load_zone_config()

    def _load_zone_config(self):
        """Đọc vùng chờ (ROI) từ DB và khởi tạo/cập nhật DwellMonitor."""
        from app.db.models import ZoneConfig
        s = get_session()
        try:
            zone = s.query(ZoneConfig).filter_by(active=True).first()
            if zone and zone.roi_points:
                roi = tuple(float(v) for v in zone.roi_points.split(","))
                thr = zone.dwell_threshold_sec or settings.DWELL_THRESHOLD_SEC
            else:
                roi = (0.1, 0.5, 0.9, 0.95)
                thr = settings.DWELL_THRESHOLD_SEC
        finally:
            s.close()
        self._roi_ratio = roi
        if self.dwell is None:
            self.dwell = DwellMonitor(roi, thr)
        else:
            self.dwell.set_roi(roi, thr)

    def _load_line_config(self, frame_w=1280, frame_h=720):
        """Đọc cấu hình vạch ảo từ DB, đổi từ tỉ lệ 0..1 sang pixel."""
        s = get_session()
        try:
            cam = s.query(CameraConfig).first()
            if cam:
                self.camera_id = cam.id
            line = s.query(LineConfig).filter_by(active=True).first()
            if line and line.line_points:
                x1, y1, x2, y2 = [float(v) for v in line.line_points.split(",")]
                rule = line.direction_rule or settings.IN_DIRECTION
            else:
                x1, y1, x2, y2 = (settings.LINE_X1, settings.LINE_Y1,
                                  settings.LINE_X2, settings.LINE_Y2)
                rule = settings.IN_DIRECTION
        finally:
            s.close()
        pts = (x1 * frame_w, y1 * frame_h, x2 * frame_w, y2 * frame_h)
        if self.counter is None:
            self.counter = LineCounter(
                pts, rule, settings.DEDUP_SECONDS, settings.LINE_HYSTERESIS_PX)
        else:
            self.counter.set_line(pts, rule)
        self._line_ratio = (x1, y1, x2, y2)

    def reload_line_config(self):
        """Áp dụng vạch theo đúng kích thước frame đang xử lý."""
        self._load_line_config(self.frame_width, self.frame_height)

    # ---------- vòng lặp xử lý ----------
    def _run(self):
        try:
            self._lazy_init()
        except Exception as e:  # noqa
            log.exception("Loi khoi tao model: %s", e)
            self._set_source_state("offline", f"Loi khoi tao AI: {e}")
            self._running = False
            return

        cap = _open_source(self.video_source)
        if not cap.isOpened():
            log.error("Khong mo duoc nguon video: %s", self.video_source)
            self._set_source_state("offline", f"Khong mo duoc {self.video_source}")
            self._raise_alert("CAMERA_OFFLINE", "",
                              f"Khong mo duoc nguon video {self.video_source}",
                              "CRIT")
            self._running = False
            return

        def _setup_caps():
            """Lấy fps/kích thước từ cap hiện tại."""
            sfps = cap.get(cv2.CAP_PROP_FPS) or 25
            self.source_fps = float(sfps)
            fs = max(1, int(round(sfps / settings.FPS_PROCESS))) if sfps > 0 else 1
            ww = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            self.frame_width, self.frame_height = ww, hh
            return fs, ww, hh

        frame_skip, w, h = _setup_caps()
        self.clip_buffer = (ClipBuffer(fps=int(settings.FPS_PROCESS))
                            if settings.ENABLE_EVIDENCE_CLIPS else None)
        self._load_line_config(w, h)
        self._set_source_state("running")

        frame_idx = 0
        t_last = time.time()
        fps_ema = 0.0
        next_frame_at = time.time()

        while self._running:
            if self._reset_requested and not self._reload_source:
                self._reset_session_state(self._reset_reason or "manual")

            # Yêu cầu đổi nguồn video (runtime) -> mở lại capture
            if self._reload_source:
                self._reload_source = False
                log.info("Doi nguon video sang: %s", self.video_source)
                cap.release()
                cap = _open_source(self.video_source)
                if not cap.isOpened():
                    self._set_source_state(
                        "offline", f"Khong mo duoc {self.video_source}")
                    self._raise_alert("CAMERA_OFFLINE", "",
                                      f"Khong mo duoc nguon moi {self.video_source}",
                                      "CRIT")
                    time.sleep(1.0)
                    self._reload_source = True
                    continue
                frame_skip, w, h = _setup_caps()
                self.clip_buffer = (ClipBuffer(fps=int(settings.FPS_PROCESS))
                                    if settings.ENABLE_EVIDENCE_CLIPS else None)
                self._load_line_config(w, h)
                self._reset_session_state("source_switch")
                self._set_source_state("running")
                frame_idx = 0
                t_last = time.time()
                fps_ema = 0.0
                next_frame_at = time.time()

            ok, frame = cap.read()
            if not ok:
                # File demo kết thúc: dừng ở frame cuối để không cộng lặp âm thầm.
                src = str(self.video_source)
                is_file = self._is_file_source()
                if is_file:
                    if settings.VIDEO_LOOP:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._reset_session_state("video_loop")
                        self._set_source_state("running")
                        frame_idx = 0
                        next_frame_at = time.time()
                    else:
                        self._set_source_state("ended")
                        time.sleep(0.1)
                    continue
                log.warning("Mat frame tu nguon video, thu lai...")
                self._set_source_state("offline", "Mat tin hieu camera")
                self._raise_alert("CAMERA_OFFLINE", "", "Mat tin hieu camera", "WARN")
                time.sleep(1.0)
                cap.release()
                cap = _open_source(self.video_source)
                if cap.isOpened():
                    frame_skip, w, h = _setup_caps()
                    self._load_line_config(w, h)
                    self._set_source_state("running")
                continue

            # File demo chạy theo FPS gốc để người xem theo dõi được, nhưng không
            # ngủ thêm nếu AI đã xử lý chậm hơn thời gian thực.
            if self._is_file_source() and settings.PLAYBACK_REALTIME:
                next_frame_at += 1.0 / max(self.source_fps, 1.0)
                delay = next_frame_at - time.time()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1.0:
                    next_frame_at = time.time()

            frame_idx += 1
            if self.clip_buffer is not None:
                self.clip_buffer.push(frame)
            if frame_idx % frame_skip != 0:
                continue

            # Bọc toàn bộ xử lý AI để 1 lỗi lẻ không làm chết cả pipeline
            try:
                tracks = self.detector.track(frame)

                # Đọc biển số liên tục cho các xe đang tiến gần vạch -> giữ kết quả
                # tốt nhất cho từng track (xe rõ nhất khi ở gần camera, không phải
                # đúng khoảnh khắc cắt vạch).
                if self.recognizer is not None:
                    self._update_track_plates(tracks, frame)

                events = self.counter.update(tracks)

                # LineCounter là nguồn sự thật của phiên demo. OCR/phân loại tải
                # có lỗi hoặc bị tắt cũng không được làm mất lượt xe.
                self.stats["in"] = self.counter.count_in
                self.stats["out"] = self.counter.count_out
                self.stats["in_yard"] = max(
                    0, self.counter.count_in - self.counter.count_out)

                for ev in events:
                    self._handle_event(ev, frame)
            except Exception as e:  # noqa
                log.exception("Loi xu ly frame (bo qua): %s", e)
                tracks = []

            # Giám sát xe chờ lâu (6.6)
            H, W = frame.shape[:2]
            waiting, dwell_alerts = self.dwell.update(tracks, W, H)
            self.waiting_vehicles = waiting
            self.stats["waiting"] = len(waiting)
            if settings.ENABLE_ANALYTICS_ALERTS:
                for da in dwell_alerts:
                    self._handle_dwell_alert(da, frame)

            # overlay + fps
            now = time.time()
            dt = now - t_last
            t_last = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt
            self.stats["fps"] = round(fps_ema, 1)

            overlay = self._draw_overlay(frame, tracks)
            with self._frame_lock:
                self._latest_frame = overlay

        cap.release()
        self._set_source_state("stopped")
        log.info("Vong lap pipeline ket thuc")

    # ---------- xử lý cảnh báo xe chờ lâu (6.6) ----------
    def _handle_dwell_alert(self, da, frame):
        # Đọc biển số của xe đang chờ để gắn vào cảnh báo
        x1, y1, x2, y2 = da["bbox"]
        H, W = frame.shape[:2]
        crop = frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)].copy()
        plate = ""
        lpr = self.recognizer.read(crop) if self.recognizer is not None and crop.size else None
        if lpr:
            plate = lpr["plate"]
        minutes = da["dwell_seconds"] // 60
        snap_url = save_snapshot(frame, plate, tag="WAIT")
        self._raise_alert(
            "LONG_WAIT", plate,
            f"Xe {plate or '(chua doc bien so)'} cho qua {minutes} phut trong vung cho",
            "WARN", snap_url)

    # ---------- xử lý 1 sự kiện cắt vạch ----------
    def _update_track_plates(self, tracks, frame):
        """Đọc biển số cho các track đang hoạt động, giữ kết quả conf cao nhất.

        Chỉ đọc xe đủ lớn (biển đủ nét) để tiết kiệm GPU. Kết quả tích luỹ vào
        self._track_plates để dùng khi xe cắt vạch.
        """
        H, W = frame.shape[:2]
        now = time.time()
        min_area = settings.LPR_MIN_BBOX_RATIO * W * H
        seen = set()
        for t in tracks:
            tid = t["track_id"]
            seen.add(tid)
            state = self._track_plates.setdefault(tid, {
                "candidates": [], "attempts": 0, "last_attempt": 0.0,
                "last_seen": now, "plate": "", "conf": 0.0,
                "plate_img": None, "votes": 0, "confirmed": False,
            })
            state["last_seen"] = now
            x1, y1, x2, y2 = t["bbox"]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area < min_area:
                continue
            if state["confirmed"] or state["attempts"] >= settings.LPR_MAX_ATTEMPTS:
                continue
            if now - state["last_attempt"] < settings.LPR_INTERVAL_SEC:
                continue
            crop = frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            if crop.size == 0:
                continue
            state["attempts"] += 1
            state["last_attempt"] = now
            res = self.recognizer.read(crop)
            if not res or not res.get("plate"):
                continue
            state["candidates"].append(res)
            state["candidates"] = state["candidates"][-settings.LPR_MAX_ATTEMPTS:]
            best = select_best_plate(state["candidates"])
            if best:
                state.update({
                    "plate": best["plate"], "conf": best["confidence"],
                    "plate_img": best.get("plate_img"), "votes": best["votes"],
                })
                state["confirmed"] = (
                    best["votes"] >= settings.LPR_MIN_VOTES and
                    best["confidence"] >= settings.LPR_GOOD_CONF)

        # Dọn cả track đã đọc thất bại; nếu không các video dài sẽ tăng bộ nhớ mãi.
        for tid, state in list(self._track_plates.items()):
            if tid not in seen and now - state.get("last_seen", 0) > settings.LPR_TRACK_TTL_SEC:
                self._track_plates.pop(tid, None)

    def _handle_event(self, ev, frame):
        x1, y1, x2, y2 = ev["bbox"]
        H, W = frame.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        crop = frame[y1c:y2c, x1c:x2c].copy()

        # Biển số: ưu tiên kết quả tốt nhất đã tích luỹ qua nhiều frame của track này.
        plate, plate_conf, plate_img = "", 0.0, None
        best = self._track_plates.get(ev["track_id"])
        if best and best.get("plate"):
            plate, plate_conf = best["plate"], best["conf"]
            plate_img = best.get("plate_img")
        # Nếu chưa có (track mới xuất hiện ngay tại vạch), đọc thử lần cuối.
        if not plate and self.recognizer is not None:
            lpr = self.recognizer.read(crop)
            if lpr:
                plate, plate_conf = lpr["plate"], lpr["confidence"]
                plate_img = lpr.get("plate_img")

        if self.recognizer is None:
            plate_status = "OCR_DISABLED"
        elif plate:
            plate_status = "OCR_OK"
        else:
            plate_status = "OCR_NOT_FOUND"

        # Chống trùng biển số là tuỳ chọn. Mặc định demo tắt vì OCR sai không
        # được phép làm mất lượt line-crossing hợp lệ.
        if settings.ENABLE_PLATE_DEDUP and plate and not self.counter.dedup_by_plate(
                plate, ev["direction"]):
            log.info("Bo qua su kien trung bien so %s %s", plate, ev["direction"])
            return

        # Phân loại tải
        if self.classifier is not None:
            cargo_type, load_status = self.classifier.classify(crop, ev["cls_name"])
        else:
            cargo_type, load_status = "UNKNOWN", "UNKNOWN"

        # Lưu bằng chứng
        snap_url = save_snapshot(frame, plate, tag=ev["direction"])
        plate_crop_url = (save_snapshot(plate_img, plate, tag="PLATE")
                          if plate_img is not None and plate_img.size else "")
        clip_url = (self.clip_buffer.save_clip(plate, tag=ev["direction"])
                    if self.clip_buffer is not None else "")

        # Ghi DB
        event_row = self._save_event(ev, plate, plate_conf, cargo_type,
                                     load_status, snap_url, clip_url,
                                     plate_status, plate_crop_url)

        # Cảnh báo biển số confidence thấp
        if settings.ENABLE_ANALYTICS_ALERTS and self.recognizer is not None and plate and plate_conf < settings.OCR_MIN_CONF:
            self._raise_alert("LOW_PLATE_CONF", plate,
                              f"Bien so {plate} do tin cay thap ({plate_conf:.2f})",
                              "WARN", snap_url)
        elif settings.ENABLE_ANALYTICS_ALERTS and self.recognizer is not None and not plate:
            self._raise_alert("NO_PLATE", "",
                              "Khong doc duoc bien so xe qua cong", "INFO", snap_url)

        # Cảnh báo xe quay đầu bất thường (6.7)
        if settings.ENABLE_ANALYTICS_ALERTS and ev.get("u_turn"):
            self._raise_alert("U_TURN", plate,
                              f"Xe {plate or '#'+str(ev['track_id'])} quay dau bat thuong tai vach",
                              "WARN", snap_url)

        # Cảnh báo 1 biển số xuất hiện ở 2 vị trí bất hợp lý (6.7)
        if settings.ENABLE_ANALYTICS_ALERTS and plate:
            self._check_plate_dup_location(plate, event_row, snap_url)

        # Thu thập ảnh crop cho việc train sau này (nếu bật)
        self._collect_training_sample(crop, ev["cls_name"], cargo_type)

        payload = {
            "type": "vehicle_event",
            "session_id": self.session_id,
            "id": event_row.id if event_row else None,
            "plate": plate or "?",
            "plate_confidence": round(plate_conf, 2),
            "plate_status": plate_status,
            "plate_crop_url": plate_crop_url,
            "direction": ev["direction"],
            "vehicle_type": ev["cls_name"],
            "cargo_type": cargo_type,
            "load_status": load_status,
            "confidence": round(ev["conf"], 2),
            "snapshot_url": snap_url,
            "clip_url": clip_url,
            "event_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stats": dict(self.stats),
        }
        self.recent_events.insert(0, payload)
        self.recent_events = self.recent_events[:50]
        self._publish(payload)

    def _check_plate_dup_location(self, plate, event_row, snap_url):
        """1 biển số xuất hiện ở camera/vị trí khác trong <60s => bất hợp lý (6.7)."""
        if event_row is None:
            return
        s = get_session()
        try:
            recent = datetime.now() - timedelta(seconds=60)
            other = (
                s.query(VehicleEvent)
                .filter(VehicleEvent.plate_number == plate,
                        VehicleEvent.id != event_row.id,
                        VehicleEvent.camera_id != self.camera_id,
                        VehicleEvent.event_time >= recent)
                .first()
            )
            if other:
                self._raise_alert(
                    "PLATE_DUP_LOC", plate,
                    f"Bien so {plate} xuat hien dong thoi o camera {other.camera_id} "
                    f"va {self.camera_id}",
                    "WARN", snap_url)
        except Exception as e:  # noqa
            log.debug("check dup loc loi: %s", e)
        finally:
            s.close()

    def _collect_training_sample(self, crop, cls_name, cargo_type):
        """Lưu ảnh crop xe để phục vụ fine-tune/train sau này (mục dữ liệu nhà máy).
        Chỉ hoạt động khi settings.COLLECT_TRAINING_DATA = True."""
        if not settings.COLLECT_TRAINING_DATA or crop is None or crop.size == 0:
            return
        try:
            import os
            from app.core.config import BASE_DIR
            day = datetime.now().strftime("%Y%m%d")
            out = os.path.join(str(BASE_DIR), "data", "training_raw", day)
            os.makedirs(out, exist_ok=True)
            ts = datetime.now().strftime("%H%M%S_%f")[:-3]
            fname = f"{cls_name}_{cargo_type}_{ts}.jpg"
            cv2.imwrite(os.path.join(out, fname), crop)
        except Exception as e:  # noqa
            log.debug("collect sample loi: %s", e)

    # ---------- DB helpers ----------
    def _save_event(self, ev, plate, plate_conf, cargo_type, load_status,
                    snap_url, clip_url, plate_status="PENDING", plate_crop_url=""):
        s = get_session()
        try:
            row = VehicleEvent(
                camera_id=self.camera_id,
                gate_id=settings.CAMERA_TYPE,
                plate_number=plate or None,
                vehicle_type=ev["cls_name"],
                direction=ev["direction"],
                cargo_type=cargo_type,
                load_status=load_status,
                confidence_score=ev["conf"],
                plate_confidence=plate_conf,
                plate_status=plate_status,
                plate_crop_url=plate_crop_url or None,
                session_id=self.session_id,
                event_time=datetime.now(),
                snapshot_url=snap_url,
                video_clip_url=clip_url,
                status="NEW",
                track_id=ev["track_id"],
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            self._try_build_trip(s, row)
            return row
        except Exception as e:  # noqa
            log.exception("Loi ghi su kien: %s", e)
            s.rollback()
            return None
        finally:
            s.close()

    def _try_build_trip(self, s, out_event):
        """Khi có sự kiện OUT, ghép với sự kiện IN gần nhất cùng biển số -> 1 chuyến."""
        if out_event.direction != "OUT" or not out_event.plate_number:
            return
        in_event = (
            s.query(VehicleEvent)
            .filter(VehicleEvent.plate_number == out_event.plate_number,
                    VehicleEvent.direction == "IN",
                    VehicleEvent.event_time <= out_event.event_time)
            .order_by(VehicleEvent.event_time.desc())
            .first()
        )
        if not in_event:
            # Xe ra nhưng không có lượt vào -> cảnh báo (2.4)
            self._raise_alert("OUT_WITHOUT_IN", out_event.plate_number,
                              f"Xe {out_event.plate_number} ra nhung khong co luot vao",
                              "WARN")
            return
        dur = (out_event.event_time - in_event.event_time).total_seconds() / 60.0
        trip = VehicleTrip(
            plate_number=out_event.plate_number,
            vehicle_type=out_event.vehicle_type,
            cargo_type=out_event.cargo_type,
            in_event_id=in_event.id,
            out_event_id=out_event.id,
            time_in=in_event.event_time,
            time_out=out_event.event_time,
            duration_minutes=round(dur, 1),
            trip_status="COMPLETE",
        )
        s.add(trip)
        s.commit()

    def _raise_alert(self, atype, plate, message, severity="INFO", evidence=""):
        self.stats["alerts"] += 1
        s = get_session()
        try:
            a = AlertEvent(alert_type=atype, plate_number=plate or None,
                           camera_id=self.camera_id, severity=severity,
                           message=message, evidence_url=evidence, status="NEW")
            s.add(a)
            s.commit()
            self._publish({
                "type": "alert",
                "alert_type": atype,
                "plate": plate or "",
                "severity": severity,
                "message": message,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:  # noqa
            log.warning("Loi ghi canh bao: %s", e)
            s.rollback()
        finally:
            s.close()

    def _publish(self, payload):
        if self.event_bus is not None:
            try:
                self.event_bus.publish(payload)
            except Exception as e:  # noqa
                log.debug("publish loi: %s", e)

    # ---------- overlay / MJPEG ----------
    def _draw_overlay(self, frame, tracks):
        img = frame.copy()
        H, W = img.shape[:2]

        # Vùng chờ (ROI) - vẽ khung mờ
        rx1, ry1, rx2, ry2 = self._roi_ratio
        zp1 = (int(rx1 * W), int(ry1 * H))
        zp2 = (int(rx2 * W), int(ry2 * H))
        cv2.rectangle(img, zp1, zp2, (255, 150, 0), 2)
        cv2.putText(img, "VUNG CHO", (zp1[0] + 5, zp1[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

        # Vạch ảo
        x1, y1, x2, y2 = self._line_ratio
        p1 = (int(x1 * W), int(y1 * H))
        p2 = (int(x2 * W), int(y2 * H))
        cv2.line(img, p1, p2, (0, 255, 255), 3)
        cv2.putText(img, "LINE", (p1[0], p1[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Tập track_id đang chờ quá ngưỡng để tô đỏ
        over_ids = {w["track_id"] for w in self.waiting_vehicles if w.get("over_threshold")}
        wait_map = {w["track_id"]: w["dwell_seconds"] for w in self.waiting_vehicles}

        for t in tracks:
            bx1, by1, bx2, by2 = t["bbox"]
            tid = t["track_id"]
            if tid in over_ids:
                color = (0, 0, 255)          # đỏ: chờ quá lâu
            elif tid in wait_map:
                color = (0, 165, 255)        # cam: đang chờ
            else:
                color = (0, 200, 0)          # xanh: bình thường
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
            label = f"#{tid} {t['cls_name']} {t['conf']:.2f}"
            if tid in wait_map:
                label += f" wait {wait_map[tid]//60}m{wait_map[tid]%60}s"
            cv2.putText(img, label, (bx1, max(15, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Bảng thống kê
        panel = f"IN:{self.stats['in']}  OUT:{self.stats['out']}  " \
                f"YARD:{self.stats['in_yard']}  CHO:{self.stats['waiting']}  " \
                f"FPS:{self.stats['fps']}"
        cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
        cv2.putText(img, panel, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    def get_jpeg(self):
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else None
