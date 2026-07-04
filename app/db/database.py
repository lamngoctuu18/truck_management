"""Kết nối DB, tạo database + bảng nếu chưa có."""
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.db.models import Base

_engine = None
_SessionLocal = None


def init_db():
    """Tạo database (nếu chưa có) và toàn bộ bảng."""
    global _engine, _SessionLocal

    # 1) Tạo database nếu chưa tồn tại (PostgreSQL: không có IF NOT EXISTS,
    #    và CREATE DATABASE phải chạy ngoài transaction -> dùng AUTOCOMMIT)
    tmp_engine = create_engine(settings.db_url_no_db, pool_pre_ping=True,
                               isolation_level="AUTOCOMMIT")
    with tmp_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"),
            {"n": settings.DB_NAME},
        ).scalar()
        if not exists:
            # Tên DB không tham số hoá được trong DDL -> escape dấu " để an toàn
            safe_name = settings.DB_NAME.replace('"', '""')
            conn.execute(text(f'CREATE DATABASE "{safe_name}" ENCODING \'UTF8\''))
    tmp_engine.dispose()

    # 2) Kết nối tới database và tạo bảng
    _engine = create_engine(settings.db_url, pool_pre_ping=True, pool_recycle=3600)
    Base.metadata.create_all(_engine)
    _migrate_vehicle_event()
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    _seed_defaults()
    return _engine


def _migrate_vehicle_event():
    """Bổ sung cột mới cho DB POC đã tồn tại (create_all không ALTER bảng)."""
    columns = {c["name"] for c in inspect(_engine).get_columns("vehicle_event")}
    # Cú pháp tương thích PostgreSQL (cột mặc định nullable, không ghi NULL inline).
    additions = {
        "plate_status": "VARCHAR(20) DEFAULT 'PENDING'",
        "plate_crop_url": "VARCHAR(255)",
        "session_id": "VARCHAR(16)",
    }
    with _engine.begin() as conn:
        for name, ddl in additions.items():
            if name not in columns:
                conn.execute(text(
                    f'ALTER TABLE vehicle_event ADD COLUMN "{name}" {ddl}'
                ))
        # Index riêng để lọc sự kiện theo phiên; bỏ qua nếu đã tồn tại.
        indexes = {i["name"] for i in inspect(_engine).get_indexes("vehicle_event")}
        if "ix_vehicle_event_session_id" not in indexes:
            conn.execute(text(
                "CREATE INDEX ix_vehicle_event_session_id ON vehicle_event (session_id)"
            ))


def get_engine():
    if _engine is None:
        init_db()
    return _engine


def get_session():
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def _seed_defaults():
    """Tạo bản ghi camera + vạch ảo mặc định nếu bảng trống."""
    from app.db.models import CameraConfig, LineConfig, ZoneConfig
    s = _SessionLocal()
    try:
        if s.query(CameraConfig).count() == 0:
            cam = CameraConfig(
                camera_name=settings.CAMERA_NAME,
                rtsp_url=str(settings.VIDEO_SOURCE),
                location="Cong chinh",
                camera_type=settings.CAMERA_TYPE,
                status="ACTIVE",
                fps_process=settings.FPS_PROCESS,
            )
            s.add(cam)
            s.commit()
            line = LineConfig(
                camera_id=cam.id,
                line_name="Vach cong",
                line_points=f"{settings.LINE_X1},{settings.LINE_Y1},"
                            f"{settings.LINE_X2},{settings.LINE_Y2}",
                direction_rule=settings.IN_DIRECTION,
                active=True,
            )
            s.add(line)
            s.commit()

        # Vùng chờ mặc định (seed riêng để áp dụng cho cả DB đã có camera từ trước)
        if s.query(ZoneConfig).count() == 0:
            cam = s.query(CameraConfig).first()
            zone = ZoneConfig(
                camera_id=cam.id if cam else 1,
                zone_name="Khu vuc cho",
                zone_type="QUEUE",
                roi_points="0.1,0.5,0.9,0.95",
                dwell_threshold_sec=settings.DWELL_THRESHOLD_SEC,
                active=True,
            )
            s.add(zone)
            s.commit()
    finally:
        s.close()
