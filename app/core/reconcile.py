"""Tích hợp cân xe / phiếu xuất kho + đối soát (Giai đoạn 6).

Chức năng:
  1. Import phiếu cân từ file CSV/Excel (nhiều tên cột linh hoạt).
  2. Đối soát mỗi phiếu với sự kiện xe (VehicleEvent) theo biển số + thời gian
     trong cửa sổ RECONCILE_TIME_WINDOW_MIN phút.
  3. Gắn trạng thái MATCHED / MISMATCH / UNMATCHED; sinh cảnh báo nếu lệch.

Ánh xạ cột (không phân biệt hoa/thường, bỏ dấu cách):
  ticket_no    <- ticket_no | so_phieu | sophieu | ticket | phieu
  plate_number <- plate | plate_number | bien_so | bienso | bks
  weight_in    <- weight_in | trong_luong_vao | tlvao | can_lan_1
  weight_out   <- weight_out | trong_luong_ra | tlra | can_lan_2
  net_weight   <- net_weight | khoi_luong | tinh | net
  cargo_type   <- cargo_type | loai_hang | hang
  ticket_time  <- time | thoi_gian | ticket_time | gio_can
  operator     <- operator | nguoi_can | nhan_vien
"""
import csv
import io
from datetime import datetime, timedelta

from app.core.config import settings
from app.core.logger import get_logger
from app.db.database import get_session
from app.db.models import WeighTicket, VehicleEvent
from app.models_ai.plate_recognizer import normalize_plate

log = get_logger("Reconcile")

_COL_ALIASES = {
    "ticket_no": ["ticket_no", "so_phieu", "sophieu", "ticket", "phieu", "so phieu"],
    "plate_number": ["plate", "plate_number", "bien_so", "bienso", "bks", "bien so"],
    "weight_in": ["weight_in", "trong_luong_vao", "tlvao", "can_lan_1", "trong luong vao"],
    "weight_out": ["weight_out", "trong_luong_ra", "tlra", "can_lan_2", "trong luong ra"],
    "net_weight": ["net_weight", "khoi_luong", "net", "tinh", "khoi luong"],
    "cargo_type": ["cargo_type", "loai_hang", "hang", "loai hang"],
    "ticket_time": ["ticket_time", "time", "thoi_gian", "gio_can", "thoi gian", "gio can"],
    "operator": ["operator", "nguoi_can", "nhan_vien", "nguoi can"],
}

_TIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
    "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S",
]


def _norm_key(k):
    return (k or "").strip().lower().replace("_", " ").replace("-", " ")


def _build_colmap(headers):
    """Trả về dict field -> index cột dựa trên alias."""
    norm = [_norm_key(h) for h in headers]
    colmap = {}
    for field, aliases in _COL_ALIASES.items():
        for i, h in enumerate(norm):
            if h in [_norm_key(a) for a in aliases]:
                colmap[field] = i
                break
    return colmap


def _parse_time(val):
    if not val:
        return None
    if isinstance(val, datetime):
        return val
    v = str(val).strip()
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            continue
    return None


def _to_float(val):
    if val is None or val == "":
        return None
    try:
        return float(str(val).replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def _rows_from_csv(content: bytes):
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _rows_from_xlsx(content: bytes):
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    ws = wb.active
    rows = [[c for c in r] for r in ws.iter_rows(values_only=True)]
    wb.close()
    if not rows:
        return [], []
    return rows[0], rows[1:]


def import_tickets(filename: str, content: bytes):
    """Import phiếu cân từ nội dung file. Trả về thống kê."""
    if filename.lower().endswith((".xlsx", ".xlsm")):
        headers, data = _rows_from_xlsx(content)
    else:
        headers, data = _rows_from_csv(content)

    if not headers:
        return {"ok": False, "error": "File rong hoac khong doc duoc"}

    colmap = _build_colmap(headers)
    if "plate_number" not in colmap:
        return {"ok": False,
                "error": f"Khong tim thay cot bien so. Cot doc duoc: {headers}"}

    s = get_session()
    imported, skipped = 0, 0
    try:
        for row in data:
            if not row or all(c in (None, "") for c in row):
                continue

            def get(field):
                idx = colmap.get(field)
                if idx is None or idx >= len(row):
                    return None
                return row[idx]

            plate = normalize_plate(str(get("plate_number") or ""))
            if not plate:
                skipped += 1
                continue
            ticket = WeighTicket(
                ticket_no=str(get("ticket_no") or "").strip() or None,
                plate_number=plate,
                weight_in=_to_float(get("weight_in")),
                weight_out=_to_float(get("weight_out")),
                net_weight=_to_float(get("net_weight")),
                cargo_type=(str(get("cargo_type")).strip().upper()
                            if get("cargo_type") else None),
                ticket_time=_parse_time(get("ticket_time")),
                operator=str(get("operator") or "").strip() or None,
                source="IMPORT",
            )
            # Tự tính net nếu thiếu
            if ticket.net_weight is None and ticket.weight_in and ticket.weight_out:
                ticket.net_weight = abs(ticket.weight_in - ticket.weight_out)
            s.add(ticket)
            imported += 1
        s.commit()
    finally:
        s.close()

    # Đối soát ngay sau khi import
    matched = reconcile_all()
    return {"ok": True, "imported": imported, "skipped": skipped,
            "matched": matched}


def reconcile_all(raise_alert_cb=None):
    """Đối soát toàn bộ phiếu chưa ghép với sự kiện xe. Trả về số phiếu vừa ghép."""
    window = timedelta(minutes=settings.RECONCILE_TIME_WINDOW_MIN)
    s = get_session()
    matched = 0
    try:
        pending = s.query(WeighTicket).filter(
            WeighTicket.match_status == "UNMATCHED").all()
        for tk in pending:
            # Tìm sự kiện xe cùng biển số, gần thời gian phiếu nhất
            q = s.query(VehicleEvent).filter(
                VehicleEvent.plate_number == tk.plate_number)
            if tk.ticket_time:
                lo = tk.ticket_time - window
                hi = tk.ticket_time + window
                q = q.filter(VehicleEvent.event_time >= lo,
                             VehicleEvent.event_time <= hi)
            ev = q.order_by(VehicleEvent.event_time.desc()).first()

            if ev is None:
                continue

            tk.matched_event_id = ev.id
            # Cập nhật trọng lượng vào sự kiện xe để báo cáo sản lượng
            ev.weight_in = tk.weight_in
            ev.weight_out = tk.weight_out
            ev.net_weight = tk.net_weight
            ev.ticket_no = tk.ticket_no

            # So khớp biển số (AI vs phiếu) trên dạng đã chuẩn hoá.
            # Nếu biển hiệu chỉnh của user khác phiếu -> MISMATCH.
            ai_plate = normalize_plate(ev.corrected_plate_number or ev.plate_number or "")
            if ai_plate and tk.plate_number and ai_plate != tk.plate_number:
                tk.match_status = "MISMATCH"
                tk.note = f"Bien so AI={ai_plate} khac phieu={tk.plate_number}"
                if raise_alert_cb:
                    raise_alert_cb("RECONCILE_MISMATCH", tk.plate_number,
                                   tk.note, "WARN", ev.snapshot_url or "")
            else:
                tk.match_status = "MATCHED"
            matched += 1
        s.commit()
    finally:
        s.close()
    log.info("Reconcile: da ghep %d phieu", matched)
    return matched
