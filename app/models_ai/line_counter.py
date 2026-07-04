"""Module đếm line-crossing (6.3).

Xác định xe cắt qua vạch ảo và hướng di chuyển (IN/OUT) dựa trên phía của tâm xe
so với đường thẳng vạch (dấu của tích có hướng - side sign).

Quy tắc:
  - Vạch định nghĩa bởi 2 điểm A(x1,y1), B(x2,y2).
  - Với tâm P, side = sign((B-A) x (P-A)).
  - Khi side đổi dấu giữa 2 frame liên tiếp -> xe đã cắt vạch.
  - direction_rule='down': đi từ side<0 sang side>0 tính là IN; ngược lại OUT.
    (side>0 ứng với phía "dưới/trong" tuỳ cách đặt điểm; có thể đảo bằng rule)
"""
import math
import time
from app.core.logger import get_logger

log = get_logger("LineCounter")


def _side(ax, ay, bx, by, px, py):
    """Dấu của tích có hướng (B-A) x (P-A)."""
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


class LineCounter:
    def __init__(self, line_points, direction_rule="down", dedup_seconds=45,
                 hysteresis_px=6):
        """line_points: (x1,y1,x2,y2) toạ độ tuyệt đối pixel."""
        self.dedup_seconds = dedup_seconds
        self.hysteresis_px = max(0.0, float(hysteresis_px))
        # track_id -> side gần nhất
        self._last_side = {}
        self._last_seen = {}
        # (plate|track, direction) -> timestamp lần ghi gần nhất (chống trùng)
        self._recent = {}
        # track_id -> (direction, timestamp) lần cross gần nhất -> phát hiện quay đầu
        self._last_cross = {}
        self.uturn_window = 20   # giây: cross ngược hướng trong khoảng này = quay đầu
        self.count_in = 0
        self.count_out = 0
        self.set_line(line_points, direction_rule)

    def set_line(self, line_points, direction_rule="down"):
        self.x1, self.y1, self.x2, self.y2 = line_points
        self.direction_rule = direction_rule
        # Không dùng vị trí cũ với vạch mới vì sẽ tạo một lượt cắt giả.
        self._last_side.clear()
        self._last_cross.clear()

    def reset(self):
        """Bắt đầu phiên đếm mới nhưng giữ nguyên cấu hình vạch."""
        self._last_side.clear()
        self._last_seen.clear()
        self._recent.clear()
        self._last_cross.clear()
        self.count_in = 0
        self.count_out = 0

    def _stable_side(self, px, py):
        """Trả về phía ổn định của tâm xe, có vùng chết chống rung quanh vạch."""
        length = math.hypot(self.x2 - self.x1, self.y2 - self.y1)
        if length < 1e-6:
            return 0
        distance = _side(self.x1, self.y1, self.x2, self.y2, px, py) / length
        if distance > self.hysteresis_px:
            return 1
        if distance < -self.hysteresis_px:
            return -1
        return 0

    def _cleanup_stale(self, now, ttl=10):
        for tid, last_seen in list(self._last_seen.items()):
            if now - last_seen > ttl:
                self._last_seen.pop(tid, None)
                self._last_side.pop(tid, None)
                self._last_cross.pop(tid, None)
        recent_ttl = max(self.dedup_seconds * 2, ttl)
        for key, ts in list(self._recent.items()):
            if now - ts > recent_ttl:
                self._recent.pop(key, None)

    def _dedup_ok(self, key):
        now = time.time()
        last = self._recent.get(key, 0)
        if now - last < self.dedup_seconds:
            return False
        self._recent[key] = now
        return True

    def update(self, tracks):
        """Nhận list track (từ VehicleDetector.track). Trả về list sự kiện cross:
        [{track_id, direction, bbox, cls_name, conf}]
        """
        events = []
        now = time.time()
        for t in tracks:
            tid = t["track_id"]
            self._last_seen[tid] = now
            cur = self._stable_side(t["cx"], t["cy"])
            prev = self._last_side.get(tid)

            # Trong vùng chết: giữ phía ổn định trước đó, chưa kết luận đã qua vạch.
            if cur == 0:
                continue
            self._last_side[tid] = cur
            if prev is None or prev == cur:
                continue

            # Đã cắt vạch: xác định hướng
            # prev=-1 -> cur=+1 : đi sang phía dương
            crossing_positive = (prev < 0 and cur > 0)
            if self.direction_rule == "down":
                direction = "IN" if crossing_positive else "OUT"
            else:  # 'up' -> đảo
                direction = "OUT" if crossing_positive else "IN"

            # Chống trùng theo track_id + hướng
            if not self._dedup_ok((tid, direction)):
                continue

            if direction == "IN":
                self.count_in += 1
            else:
                self.count_out += 1

            # Phát hiện quay đầu: cùng track cắt vạch theo hướng ngược lại
            # trong khoảng thời gian ngắn (6.7)
            u_turn = False
            prev_cross = self._last_cross.get(tid)
            if prev_cross and prev_cross[0] != direction and \
                    now - prev_cross[1] <= self.uturn_window:
                u_turn = True
                log.info("QUAY DAU track=%s (%s -> %s)", tid, prev_cross[0], direction)
            self._last_cross[tid] = (direction, now)

            events.append({
                "track_id": tid,
                "direction": direction,
                "bbox": t["bbox"],
                "cls_name": t["cls_name"],
                "conf": t["conf"],
                "u_turn": u_turn,
            })
            log.info("CROSS track=%s dir=%s (IN=%d OUT=%d)",
                     tid, direction, self.count_in, self.count_out)
        self._cleanup_stale(now)
        return events

    def dedup_by_plate(self, plate, direction):
        """Chống trùng bổ sung theo biển số + hướng (6.3)."""
        if not plate:
            return True
        return self._dedup_ok((plate, direction))
