"""Định nghĩa bảng dữ liệu theo mục 7 của kế hoạch triển khai."""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean, BigInteger
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class VehicleEvent(Base):
    """7.1 - Sự kiện xe vào/ra."""
    __tablename__ = "vehicle_event"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    camera_id = Column(Integer, index=True)
    gate_id = Column(String(50))
    plate_number = Column(String(32), index=True)            # biển số AI đọc
    corrected_plate_number = Column(String(32))              # biển số sau hiệu chỉnh
    vehicle_type = Column(String(32))                        # truck/ben/container...
    direction = Column(String(8), index=True)                # IN / OUT
    cargo_type = Column(String(16), default="UNKNOWN")       # SOIL/BRICK/EMPTY/UNKNOWN
    load_status = Column(String(16), default="UNKNOWN")      # LOADED/EMPTY/UNKNOWN
    confidence_score = Column(Float, default=0.0)            # độ tin cậy nhận diện
    plate_confidence = Column(Float, default=0.0)            # độ tin cậy OCR
    plate_status = Column(String(20), default="PENDING")     # OCR_OK/NOT_FOUND/DISABLED
    plate_crop_url = Column(String(255))                     # ảnh crop biển số tốt nhất
    session_id = Column(String(16), index=True)              # phiên demo sinh sự kiện
    event_time = Column(DateTime, default=datetime.now, index=True)
    snapshot_url = Column(String(255))
    video_clip_url = Column(String(255))
    weight_in = Column(Float)
    weight_out = Column(Float)
    net_weight = Column(Float)
    ticket_no = Column(String(64))
    status = Column(String(24), default="NEW")               # NEW/VERIFIED/...
    track_id = Column(Integer)                               # ID tracking tạm thời
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class CameraConfig(Base):
    """7.2 - Cấu hình camera."""
    __tablename__ = "camera_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_name = Column(String(100))
    rtsp_url = Column(String(255))
    location = Column(String(150))
    camera_type = Column(String(24))       # GATE/GATE_IN/GATE_OUT/WEIGHBRIDGE/YARD/QUEUE
    status = Column(String(12), default="ACTIVE")   # ACTIVE/INACTIVE
    fps_process = Column(Float, default=10)
    created_at = Column(DateTime, default=datetime.now)


class LineConfig(Base):
    """7.3 - Cấu hình vạch ảo line-crossing."""
    __tablename__ = "line_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(Integer, index=True)
    line_name = Column(String(80))
    line_points = Column(String(120))       # "x1,y1,x2,y2" theo tỉ lệ 0..1
    direction_rule = Column(String(20))     # "down"=IN | "up"=IN
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)


class ZoneConfig(Base):
    """Cấu hình vùng giám sát (vùng chờ - 6.6). ROI theo tỉ lệ 0..1."""
    __tablename__ = "zone_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(Integer, index=True)
    zone_name = Column(String(80))
    zone_type = Column(String(20), default="QUEUE")   # QUEUE (vùng chờ)
    roi_points = Column(String(120))                  # "x1,y1,x2,y2" tỉ lệ 0..1
    dwell_threshold_sec = Column(Integer, default=1800)  # ngưỡng chờ (giây)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)


class AlertEvent(Base):
    """7.4 - Cảnh báo."""
    __tablename__ = "alert_event"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    alert_type = Column(String(40), index=True)
    plate_number = Column(String(32))
    camera_id = Column(Integer)
    severity = Column(String(12), default="INFO")   # INFO/WARN/CRIT
    message = Column(Text)
    evidence_url = Column(String(255))
    status = Column(String(16), default="NEW")      # NEW/PROCESSING/DONE
    created_at = Column(DateTime, default=datetime.now, index=True)


class WeighTicket(Base):
    """Phiếu cân / phiếu xuất kho (Giai đoạn 6 - tích hợp cân xe).

    Nạp từ file CSV/Excel do phần mềm cân xuất, hoặc nhập tay trên dashboard.
    Sau đó đối soát với sự kiện xe (VehicleEvent) theo biển số + thời gian.
    """
    __tablename__ = "weigh_ticket"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticket_no = Column(String(64), index=True)        # số phiếu cân/xuất
    plate_number = Column(String(32), index=True)     # biển số trên phiếu
    weight_in = Column(Float)                          # trọng lượng vào (kg)
    weight_out = Column(Float)                         # trọng lượng ra (kg)
    net_weight = Column(Float)                         # khối lượng hàng (kg)
    cargo_type = Column(String(16))                    # SOIL/BRICK/... (nếu có)
    ticket_time = Column(DateTime, index=True)         # thời gian cân
    operator = Column(String(80))                      # người vận hành cân
    source = Column(String(16), default="IMPORT")      # IMPORT/MANUAL
    matched_event_id = Column(BigInteger)              # sự kiện xe đã ghép
    match_status = Column(String(20), default="UNMATCHED")  # MATCHED/UNMATCHED/MISMATCH
    note = Column(String(255))
    created_at = Column(DateTime, default=datetime.now)


class VehicleTrip(Base):
    """7.5 - Chuyến xe hoàn chỉnh (ghép IN + OUT)."""
    __tablename__ = "vehicle_trip"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    plate_number = Column(String(32), index=True)
    vehicle_type = Column(String(32))
    cargo_type = Column(String(16))
    in_event_id = Column(BigInteger)
    out_event_id = Column(BigInteger)
    time_in = Column(DateTime)
    time_out = Column(DateTime)
    duration_minutes = Column(Float)
    weight_in = Column(Float)
    weight_out = Column(Float)
    net_weight = Column(Float)
    ticket_no = Column(String(64))
    trip_status = Column(String(20), default="COMPLETE")   # COMPLETE/ABNORMAL
    created_at = Column(DateTime, default=datetime.now)
