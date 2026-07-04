"""Kết nối DB, tạo database + bảng nếu chưa có.

Chế độ chịu lỗi: nếu KHÔNG kết nối được PostgreSQL, hệ thống vẫn chạy ở chế độ
"không-DB" (detect + track + đếm + xem realtime), chỉ không lưu lịch sử. Nhờ vậy
người test chỉ cần install.bat + run.bat, không bắt buộc cài PostgreSQL.
"""
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.logger import get_logger
from app.db.models import Base

log = get_logger("DB")

_engine = None
_SessionLocal = None
db_available = False   # True nếu kết nối DB thành công; toàn hệ thống đọc cờ này


def init_db():
    """Thử tạo database + bảng. Trả về True nếu OK, False nếu không có DB.

    KHÔNG ném lỗi ra ngoài -> app không crash khi thiếu PostgreSQL.
    """
    global _engine, _SessionLocal, db_available
    try:
        # 1) Tạo database nếu chưa tồn tại (PostgreSQL: không có IF NOT EXISTS,
        #    và CREATE DATABASE phải chạy ngoài transaction -> dùng AUTOCOMMIT)
        tmp_engine = create_engine(settings.db_url_no_db, pool_pre_ping=True,
                                   isolation_level="AUTOCOMMIT",
                                   connect_args={"connect_timeout": 5})
        with tmp_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"),
                {"n": settings.DB_NAME},
            ).scalar()
            if not exists:
                safe_name = settings.DB_NAME.replace('"', '""')
                conn.execute(text(f'CREATE DATABASE "{safe_name}" ENCODING \'UTF8\''))
        tmp_engine.dispose()

        # 2) Kết nối tới database và tạo bảng
        _engine = create_engine(settings.db_url, pool_pre_ping=True,
                                pool_recycle=3600,
                                connect_args={"connect_timeout": 5})
        Base.metadata.create_all(_engine)
        _migrate_vehicle_event()
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
        _seed_defaults()
        db_available = True
        log.info("Ket noi PostgreSQL OK - luu lich su day du.")
        return True
    except Exception as e:  # noqa
        db_available = False
        _engine = None
        _SessionLocal = None
        log.warning("=" * 60)
        log.warning("KHONG ket noi duoc PostgreSQL: %s", str(e).splitlines()[0][:100])
        log.warning("-> Chay che do KHONG-DB: van dem xe + xem realtime,")
        log.warning("   nhung KHONG luu lich su. De luu, hay chay PostgreSQL.")
        log.warning("=" * 60)
        return False


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


class _NullQuery:
    """Query giả cho chế độ không-DB: mọi thao tác trả rỗng, không lỗi."""
    def filter(self, *a, **k): return self
    def filter_by(self, *a, **k): return self
    def order_by(self, *a, **k): return self
    def group_by(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def offset(self, *a, **k): return self
    def join(self, *a, **k): return self
    def all(self): return []
    def first(self): return None
    def scalar(self): return None
    def count(self): return 0
    def get(self, *a, **k): return None
    def one_or_none(self): return None


class _NullSession:
    """Session giả khi không có DB: nuốt mọi thao tác ghi/đọc, không ném lỗi.

    Nhờ đó code dùng session chạy bình thường; dữ liệu chỉ không được lưu.
    """
    def query(self, *a, **k): return _NullQuery()
    def add(self, *a, **k): pass
    def delete(self, *a, **k): pass
    def commit(self): pass
    def rollback(self): pass
    def refresh(self, *a, **k): pass
    def flush(self, *a, **k): pass
    def execute(self, *a, **k): return _NullQuery()
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


def get_engine():
    return _engine


def get_session():
    """Trả về session thật, hoặc session giả (_NullSession) nếu không có DB.

    Session giả cho phép code chạy nguyên vẹn ở chế độ không-DB mà không cần
    sửa từng chỗ gọi. Dữ liệu đơn giản là không được lưu.
    """
    if _SessionLocal is None:
        return _NullSession()
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
