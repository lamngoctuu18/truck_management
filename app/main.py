"""Backend FastAPI + WebSocket + MJPEG stream cho hệ thống AI đếm xe."""
import os
import asyncio
import math
from datetime import datetime, date

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import func, and_

from app.core.config import settings, BASE_DIR
from app.core.logger import get_logger
from app.db.database import init_db, get_session
from app.db.models import VehicleEvent, VehicleTrip, AlertEvent, CameraConfig, LineConfig
from app.api.event_bus import EventBus
from app.core.pipeline import AnalyticsPipeline

log = get_logger("Main")

# Nhóm API theo chức năng để Swagger UI dễ đọc
tags_metadata = [
    {"name": "Hệ thống", "description": "Trạng thái pipeline, phiên đếm, thông tin chung."},
    {"name": "Nguồn video", "description": "Đổi/tải nguồn video, phát lại (không cần restart)."},
    {"name": "Sự kiện xe", "description": "Truy vấn lượt xe IN/OUT, hiệu chỉnh & đọc lại biển số."},
    {"name": "Chuyến xe", "description": "Chuyến hoàn chỉnh ghép lượt vào + ra."},
    {"name": "Xe chờ lâu", "description": "Xe đứng yên quá lâu trong vùng chờ (realtime)."},
    {"name": "Cảnh báo", "description": "Cảnh báo bất thường và xử lý."},
    {"name": "Cân xe & đối soát", "description": "Nạp phiếu cân CSV/Excel, đối soát với sự kiện xe."},
    {"name": "Báo cáo", "description": "Tổng hợp số liệu theo ngày."},
    {"name": "Cấu hình", "description": "Vạch ảo line-crossing và vùng chờ."},
]

app = FastAPI(
    title="AI Đếm xe chở đất/gạch vào–ra nhà máy",
    description=(
        "REST API cho hệ thống Camera Analytics: nhận diện xe (YOLOv8) → tracking "
        "(ByteTrack) → đếm line-crossing IN/OUT → đọc biển số → đối soát cân xe.\n\n"
        "- **Tài liệu tương tác**: `/docs` (Swagger UI) · `/redoc` (ReDoc)\n"
        "- **OpenAPI JSON**: `/openapi.json`\n"
        "- **Video trực tiếp**: `/video_feed` (MJPEG) · **Realtime**: `/ws` (WebSocket)"
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
    contact={"name": "Nhóm dự án", "email": "lamngoctuk55@gmail.com"},
)

# Static + templates
app.mount("/media", StaticFiles(directory=str(BASE_DIR / "data")), name="media")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")

# Dùng Jinja2 Environment trực tiếp (tránh bug cache của Starlette+Jinja hiện tại)
_jinja = Environment(
    loader=FileSystemLoader(str(BASE_DIR / "web" / "templates")),
    autoescape=select_autoescape(["html", "xml"]),
)


def render(name: str, **ctx) -> HTMLResponse:
    html = _jinja.get_template(name).render(**ctx)
    return HTMLResponse(html)

event_bus = EventBus()
pipeline = AnalyticsPipeline(event_bus=event_bus)


@app.on_event("startup")
async def _startup():
    ok = init_db()   # không crash nếu thiếu DB -> chế độ không-DB
    event_bus.bind_loop(asyncio.get_running_loop())
    pipeline.start()
    mode = "co luu lich su (PostgreSQL)" if ok else "KHONG-DB (khong luu lich su)"
    log.info("Startup xong [%s]. Dashboard: http://%s:%s",
             mode, settings.WEB_HOST, settings.WEB_PORT)


@app.on_event("shutdown")
async def _shutdown():
    pipeline.stop()


# ==================== TRANG HTML ====================
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return render("dashboard.html")


@app.get("/events", response_class=HTMLResponse, include_in_schema=False)
async def events_page():
    return render("events.html")


@app.get("/trips", response_class=HTMLResponse, include_in_schema=False)
async def trips_page():
    return render("trips.html")


@app.get("/waiting", response_class=HTMLResponse, include_in_schema=False)
async def waiting_page():
    return render("waiting.html")


@app.get("/alerts", response_class=HTMLResponse, include_in_schema=False)
async def alerts_page():
    return render("alerts.html")


@app.get("/reconcile", response_class=HTMLResponse, include_in_schema=False)
async def reconcile_page():
    return render("reconcile.html")


@app.get("/reports", response_class=HTMLResponse, include_in_schema=False)
async def reports_page():
    return render("reports.html")


@app.get("/config", response_class=HTMLResponse, include_in_schema=False)
async def config_page():
    return render("config.html")


# ==================== VIDEO STREAM (MJPEG) ====================
def _mjpeg_generator():
    import time
    boundary = b"--frame"
    while True:
        jpg = pipeline.get_jpeg()
        if jpg is None:
            time.sleep(0.1)
            continue
        yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
               + jpg + b"\r\n")
        time.sleep(0.04)


@app.get("/video_feed", include_in_schema=False)
async def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ==================== WEBSOCKET REALTIME ====================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    q = event_bus.register()
    try:
        # gửi trạng thái ban đầu
        await ws.send_json({"type": "hello", "stats": pipeline.stats,
                            "pipeline_running": pipeline.is_running(),
                            "session": pipeline.session_status()})
        while True:
            payload = await q.get()
            await ws.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa
        log.debug("ws loi: %s", e)
    finally:
        event_bus.unregister(q)


# ==================== REST API ====================
@app.get("/api/status", tags=["Hệ thống"], summary="Trạng thái hệ thống")
async def api_status():
    import app.db.database as _db
    return {
        "pipeline_running": pipeline.is_running(),
        "stats": pipeline.stats,
        "video_source": str(pipeline.video_source),
        "device": settings.DEVICE,
        "demo_mode": settings.DEMO_MODE,
        "db_available": _db.db_available,
        "features": {
            "plate_recognition": settings.ENABLE_PLATE_RECOGNITION,
            "load_classification": settings.ENABLE_LOAD_CLASSIFICATION,
            "evidence_clips": settings.ENABLE_EVIDENCE_CLIPS,
            "analytics_alerts": settings.ENABLE_ANALYTICS_ALERTS,
        },
        "session": pipeline.session_status(),
    }


@app.get("/api/session", tags=["Hệ thống"], summary="Trạng thái phiên đếm")
async def api_session():
    """Trạng thái phiên đếm đang hiển thị trên dashboard demo."""
    return pipeline.session_status()


@app.post("/api/session/reset", tags=["Hệ thống"], summary="Reset bộ đếm phiên")
async def api_session_reset():
    """Reset bộ đếm/tracking của phiên, không xoá lịch sử database."""
    pipeline.request_reset("manual")
    return {"ok": True, "message": "Da yeu cau reset phien dem"}


@app.post("/api/video/replay", tags=["Nguồn video"], summary="Phát lại video từ đầu")
async def api_video_replay():
    """Phát nguồn hiện tại lại từ đầu bằng một phiên đếm mới."""
    pipeline.replay_source()
    return {"ok": True, "source": str(pipeline.video_source)}


# ==================== NGUỒN VIDEO (đổi khi đang chạy) ====================
@app.get("/api/video/sources", tags=["Nguồn video"], summary="Danh sách video có sẵn")
async def api_video_sources():
    """Liệt kê video có sẵn trong data/videos + nguồn hiện tại."""
    videos_dir = BASE_DIR / "data" / "videos"
    files = []
    if videos_dir.exists():
        for f in sorted(videos_dir.glob("*")):
            if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                files.append({"name": f.name,
                              "path": f"data/videos/{f.name}",
                              "size_mb": round(f.stat().st_size / 1e6, 1)})
    return {"current": str(pipeline.video_source), "videos": files}


@app.post("/api/video/upload", tags=["Nguồn video"], summary="Tải video lên & dùng ngay")
async def api_video_upload(file: UploadFile = File(...)):
    """Upload file video vào data/videos rồi chuyển sang dùng ngay."""
    name = os.path.basename(file.filename or "upload.mp4")
    if not name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        return JSONResponse({"ok": False, "error": "Chi ho tro mp4/avi/mov/mkv"},
                            status_code=400)
    videos_dir = BASE_DIR / "data" / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    dst = videos_dir / name
    content = await file.read()
    with open(dst, "wb") as f:
        f.write(content)
    rel = f"data/videos/{name}"
    pipeline.switch_source(rel)
    return {"ok": True, "path": rel, "size_mb": round(len(content) / 1e6, 1),
            "session_reset": True}


@app.post("/api/video/switch", tags=["Nguồn video"], summary="Đổi nguồn video (file/webcam/RTSP)")
async def api_video_switch(request: Request):
    """Đổi nguồn video: file có sẵn, webcam (0/1...), hoặc URL rtsp://."""
    b = await request.json()
    src = (b.get("source") or "").strip()
    if not src:
        return JSONResponse({"ok": False, "error": "Thieu source"}, status_code=400)
    pipeline.switch_source(src)
    return {"ok": True, "source": src, "session_reset": True}


@app.get("/api/waiting", tags=["Xe chờ lâu"], summary="Xe đang chờ trong vùng giám sát")
async def api_waiting():
    """Danh sách xe đang chờ trong vùng giám sát (realtime, từ pipeline)."""
    rows = []
    for w in pipeline.waiting_vehicles:
        secs = w["dwell_seconds"]
        rows.append({
            "track_id": w["track_id"],
            "vehicle_type": w["cls_name"],
            "dwell_seconds": secs,
            "dwell_text": f"{secs // 60} phút {secs % 60} giây",
            "over_threshold": w.get("over_threshold", False),
        })
    rows.sort(key=lambda x: -x["dwell_seconds"])
    return {"count": len(rows), "vehicles": rows}


@app.get("/api/config/zone", tags=["Cấu hình"], summary="Lấy cấu hình vùng chờ")
async def api_get_zone():
    from app.db.models import ZoneConfig
    s = get_session()
    try:
        z = s.query(ZoneConfig).filter_by(active=True).first()
        return {
            "roi_points": z.roi_points if z else "0.1,0.5,0.9,0.95",
            "dwell_threshold_sec": z.dwell_threshold_sec if z else 1800,
            "zone_name": z.zone_name if z else "Khu vuc cho",
        }
    finally:
        s.close()


@app.post("/api/config/zone", tags=["Cấu hình"], summary="Cập nhật vùng chờ + ngưỡng")
async def api_set_zone(request: Request):
    """Cấu hình vùng chờ + ngưỡng thời gian (mục cấu hình 6.8)."""
    from app.db.models import ZoneConfig, CameraConfig
    body = await request.json()
    roi = body.get("roi_points")
    thr = body.get("dwell_threshold_sec")
    s = get_session()
    try:
        z = s.query(ZoneConfig).filter_by(active=True).first()
        if not z:
            cam = s.query(CameraConfig).first()
            z = ZoneConfig(camera_id=cam.id if cam else 1, zone_name="Khu vuc cho",
                           zone_type="QUEUE", active=True)
            s.add(z)
        if roi:
            z.roi_points = roi
        if thr is not None:
            z.dwell_threshold_sec = int(thr)
        s.commit()
        # áp dụng ngay cho pipeline
        try:
            r = tuple(float(v) for v in z.roi_points.split(","))
            pipeline._roi_ratio = r
            if pipeline.dwell is not None:
                pipeline.dwell.set_roi(r, z.dwell_threshold_sec)
        except Exception:  # noqa
            pass
        return {"ok": True, "roi_points": z.roi_points,
                "dwell_threshold_sec": z.dwell_threshold_sec}
    finally:
        s.close()


@app.get("/api/summary", tags=["Báo cáo"], summary="Tổng quan hôm nay")
async def api_summary():
    """Tổng quan hôm nay (mục dashboard 6.8)."""
    s = get_session()
    try:
        today = date.today()
        q = s.query(VehicleEvent).filter(func.date(VehicleEvent.event_time) == today)
        total_in = q.filter(VehicleEvent.direction == "IN").count()
        total_out = q.filter(VehicleEvent.direction == "OUT").count()
        soil = q.filter(VehicleEvent.cargo_type == "SOIL").count()
        brick = q.filter(VehicleEvent.cargo_type == "BRICK").count()
        alerts = s.query(AlertEvent).filter(
            func.date(AlertEvent.created_at) == today,
            AlertEvent.status == "NEW").count()
        return {
            "date": str(today),
            "in": total_in, "out": total_out,
            "in_yard": max(0, total_in - total_out),
            "soil_trips": soil, "brick_trips": brick,
            "alerts": alerts,
            "live": pipeline.stats,
        }
    finally:
        s.close()


@app.get("/api/events", tags=["Sự kiện xe"], summary="Danh sách sự kiện xe IN/OUT")
async def api_events(limit: int = 100, plate: str = "", direction: str = "",
                     cargo: str = "", session: str = ""):
    s = get_session()
    try:
        q = s.query(VehicleEvent).order_by(VehicleEvent.event_time.desc())
        if plate:
            q = q.filter(
                (VehicleEvent.plate_number.like(f"%{plate}%")) |
                (VehicleEvent.corrected_plate_number.like(f"%{plate}%")))
        if direction:
            q = q.filter(VehicleEvent.direction == direction)
        if cargo:
            q = q.filter(VehicleEvent.cargo_type == cargo)
        if session:
            q = q.filter(VehicleEvent.session_id == session)
        rows = q.limit(limit).all()
        return [_event_dict(r) for r in rows]
    finally:
        s.close()


@app.post("/api/events/{event_id}/plate", tags=["Sự kiện xe"], summary="Hiệu chỉnh biển số")
async def api_correct_plate(event_id: int, request: Request):
    """Hiệu chỉnh biển số (6.4 - cho phép người vận hành sửa)."""
    body = await request.json()
    from app.models_ai.plate_recognizer import normalize_plate
    new_plate = normalize_plate(body.get("plate") or "")
    if not new_plate:
        return JSONResponse({"ok": False, "error": "Bien so khong hop le"},
                            status_code=400)
    s = get_session()
    try:
        row = s.query(VehicleEvent).get(event_id)
        if not row:
            return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
        row.corrected_plate_number = new_plate
        row.plate_status = "MANUAL"
        row.status = "VERIFIED"
        s.commit()
        return {"ok": True, "id": event_id, "corrected_plate_number": new_plate}
    finally:
        s.close()


@app.post("/api/events/{event_id}/reread", tags=["Sự kiện xe"], summary="Đọc lại biển số 1 sự kiện")
async def api_reread_plate(event_id: int):
    """Chạy lại ALPR trên snapshot của một sự kiện."""
    from app.core.plate_backfill import reread_event_plate
    result = reread_event_plate(event_id, pipeline.recognizer)
    if result.get("ok"):
        event_bus.publish({"type": "plate_update", **result})
        return result
    return JSONResponse(result, status_code=422 if result.get("error") != "event not found" else 404)


@app.post("/api/events/reread-missing", tags=["Sự kiện xe"], summary="Đọc lại 50 biển thiếu")
async def api_reread_missing(limit: int = Query(50, ge=1, le=200)):
    """Đọc lại tối đa N sự kiện chưa có biển, dùng chung một recognizer."""
    from app.core.plate_backfill import reread_event_plate
    from app.models_ai.plate_recognizer import get_recognizer, is_valid_plate
    s = get_session()
    try:
        candidates = (
            s.query(VehicleEvent)
            .filter(VehicleEvent.snapshot_url.isnot(None),
                    ((VehicleEvent.plate_status != "MANUAL") |
                     VehicleEvent.plate_status.is_(None)))
            .order_by(VehicleEvent.event_time.desc())
            .limit(limit * 4).all()
        )
        ids = [r.id for r in candidates
               if (not is_valid_plate(r.plate_number or "") or
                   (r.plate_confidence or 0) < settings.OCR_MIN_CONF or
                   r.plate_status != "OCR_OK")][:limit]
    finally:
        s.close()
    recognizer = pipeline.recognizer or get_recognizer()
    results = [reread_event_plate(event_id, recognizer) for event_id in ids]
    matched = sum(1 for r in results if r.get("ok"))
    event_bus.publish({"type": "plate_backfill_done", "processed": len(results),
                       "matched": matched})
    return {"ok": True, "processed": len(results), "matched": matched,
            "failed": len(results) - matched, "results": results}


@app.get("/api/trips", tags=["Chuyến xe"], summary="Danh sách chuyến xe hoàn chỉnh")
async def api_trips(limit: int = 100):
    s = get_session()
    try:
        rows = s.query(VehicleTrip).order_by(VehicleTrip.created_at.desc()).limit(limit).all()
        return [{
            "id": r.id, "plate_number": r.plate_number,
            "vehicle_type": r.vehicle_type, "cargo_type": r.cargo_type,
            "time_in": _fmt(r.time_in), "time_out": _fmt(r.time_out),
            "duration_minutes": r.duration_minutes,
            "net_weight": r.net_weight, "ticket_no": r.ticket_no,
            "trip_status": r.trip_status,
        } for r in rows]
    finally:
        s.close()


@app.get("/api/alerts", tags=["Cảnh báo"], summary="Danh sách cảnh báo")
async def api_alerts(limit: int = 100, status: str = ""):
    s = get_session()
    try:
        q = s.query(AlertEvent).order_by(AlertEvent.created_at.desc())
        if status:
            q = q.filter(AlertEvent.status == status)
        rows = q.limit(limit).all()
        return [{
            "id": r.id, "alert_type": r.alert_type, "plate_number": r.plate_number,
            "severity": r.severity, "message": r.message,
            "evidence_url": r.evidence_url, "status": r.status,
            "created_at": _fmt(r.created_at),
        } for r in rows]
    finally:
        s.close()


@app.post("/api/alerts/{alert_id}/done", tags=["Cảnh báo"], summary="Đánh dấu đã xử lý")
async def api_alert_done(alert_id: int):
    s = get_session()
    try:
        r = s.query(AlertEvent).get(alert_id)
        if not r:
            return JSONResponse({"ok": False}, status_code=404)
        r.status = "DONE"
        s.commit()
        return {"ok": True}
    finally:
        s.close()


@app.get("/api/report", tags=["Báo cáo"], summary="Báo cáo theo ngày")
async def api_report(day: str = ""):
    """Báo cáo theo ngày, gom theo hướng + loại hàng (mục 2.5)."""
    s = get_session()
    try:
        target = datetime.strptime(day, "%Y-%m-%d").date() if day else date.today()
        rows = s.query(
            VehicleEvent.direction, VehicleEvent.cargo_type,
            func.count(VehicleEvent.id)
        ).filter(func.date(VehicleEvent.event_time) == target) \
         .group_by(VehicleEvent.direction, VehicleEvent.cargo_type).all()
        table = [{"direction": d, "cargo_type": c, "count": n} for d, c, n in rows]
        total = sum(x["count"] for x in table)
        return {"date": str(target), "rows": table, "total": total}
    finally:
        s.close()


# ==================== TÍCH HỢP CÂN XE (Giai đoạn 6) ====================
@app.post("/api/weigh/import", tags=["Cân xe & đối soát"], summary="Nạp phiếu cân CSV/Excel")
async def api_weigh_import(file: UploadFile = File(...)):
    """Import phiếu cân từ file CSV/Excel + đối soát ngay."""
    from app.core.reconcile import import_tickets
    content = await file.read()
    try:
        result = import_tickets(file.filename or "upload.csv", content)
    except Exception as e:  # noqa
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return result


@app.post("/api/weigh/manual", tags=["Cân xe & đối soát"], summary="Nhập tay 1 phiếu cân")
async def api_weigh_manual(request: Request):
    """Nhập tay 1 phiếu cân trên dashboard."""
    from app.db.models import WeighTicket
    from app.core.reconcile import reconcile_all
    from app.models_ai.plate_recognizer import normalize_plate
    b = await request.json()
    plate = normalize_plate(b.get("plate_number") or "")
    if not plate:
        return JSONResponse({"ok": False, "error": "Thieu bien so"}, status_code=400)
    s = get_session()
    try:
        wi = b.get("weight_in")
        wo = b.get("weight_out")
        net = b.get("net_weight")
        if net is None and wi and wo:
            net = abs(float(wi) - float(wo))
        tt = None
        if b.get("ticket_time"):
            try:
                tt = datetime.strptime(b["ticket_time"], "%Y-%m-%dT%H:%M")
            except ValueError:
                tt = None
        tk = WeighTicket(
            ticket_no=(b.get("ticket_no") or "").strip() or None,
            plate_number=plate,
            weight_in=float(wi) if wi else None,
            weight_out=float(wo) if wo else None,
            net_weight=float(net) if net else None,
            cargo_type=(b.get("cargo_type") or "").strip().upper() or None,
            ticket_time=tt or datetime.now(),
            operator=(b.get("operator") or "").strip() or None,
            source="MANUAL",
        )
        s.add(tk)
        s.commit()
    finally:
        s.close()
    reconcile_all(pipeline._raise_alert)
    return {"ok": True}


@app.post("/api/weigh/reconcile", tags=["Cân xe & đối soát"], summary="Chạy lại đối soát")
async def api_weigh_reconcile():
    """Chạy lại đối soát toàn bộ phiếu chưa ghép."""
    from app.core.reconcile import reconcile_all
    n = reconcile_all(pipeline._raise_alert)
    return {"ok": True, "matched": n}


@app.get("/api/weigh/tickets", tags=["Cân xe & đối soát"], summary="Danh sách phiếu cân")
async def api_weigh_tickets(limit: int = 200, status: str = ""):
    from app.db.models import WeighTicket
    s = get_session()
    try:
        q = s.query(WeighTicket).order_by(WeighTicket.created_at.desc())
        if status:
            q = q.filter(WeighTicket.match_status == status)
        rows = q.limit(limit).all()
        return [{
            "id": r.id, "ticket_no": r.ticket_no, "plate_number": r.plate_number,
            "weight_in": r.weight_in, "weight_out": r.weight_out,
            "net_weight": r.net_weight, "cargo_type": r.cargo_type,
            "ticket_time": _fmt(r.ticket_time), "operator": r.operator,
            "source": r.source, "match_status": r.match_status,
            "matched_event_id": r.matched_event_id, "note": r.note,
        } for r in rows]
    finally:
        s.close()


# ==================== CONFIG API ====================
@app.get("/api/config/line", tags=["Cấu hình"], summary="Lấy cấu hình vạch ảo")
async def api_get_line():
    s = get_session()
    try:
        line = s.query(LineConfig).filter_by(active=True).first()
        cam = s.query(CameraConfig).first()
        return {
            "line_points": line.line_points if line else "0.05,0.55,0.95,0.55",
            "direction_rule": line.direction_rule if line else "down",
            "camera_name": cam.camera_name if cam else "",
            "camera_type": cam.camera_type if cam else "GATE",
            "rtsp_url": cam.rtsp_url if cam else "",
        }
    finally:
        s.close()


@app.post("/api/config/line", tags=["Cấu hình"], summary="Cập nhật vạch ảo line-crossing")
async def api_set_line(request: Request):
    """Cấu hình vạch ảo (mục cấu hình dashboard 6.8)."""
    body = await request.json()
    pts = body.get("line_points")     # "x1,y1,x2,y2" 0..1
    rule = body.get("direction_rule", "down")
    try:
        values = [float(v) for v in str(pts).split(",")]
        if len(values) != 4 or not all(math.isfinite(v) and 0 <= v <= 1
                                       for v in values):
            raise ValueError
        if values[0] == values[2] and values[1] == values[3]:
            raise ValueError
    except (TypeError, ValueError):
        return JSONResponse(
            {"ok": False, "error": "line_points phai gom 4 so 0..1 va 2 diem khac nhau"},
            status_code=400)
    if rule not in {"down", "up"}:
        return JSONResponse({"ok": False, "error": "direction_rule khong hop le"},
                            status_code=400)
    pts = ",".join(f"{v:.6f}" for v in values)
    s = get_session()
    try:
        line = s.query(LineConfig).filter_by(active=True).first()
        if not line:
            cam = s.query(CameraConfig).first()
            line = LineConfig(camera_id=cam.id if cam else 1, line_name="Vach cong",
                              active=True)
            s.add(line)
        if pts:
            line.line_points = pts
        line.direction_rule = rule
        s.commit()
        # cập nhật ngay cho pipeline
        try:
            pipeline.reload_line_config()
        except Exception:  # noqa
            pass
        return {"ok": True, "line_points": line.line_points,
                "direction_rule": line.direction_rule}
    finally:
        s.close()


# ==================== helpers ====================
def _fmt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None


def _event_dict(r: VehicleEvent):
    return {
        "id": r.id,
        "camera_id": r.camera_id,
        "gate_id": r.gate_id,
        "plate_number": r.plate_number,
        "corrected_plate_number": r.corrected_plate_number,
        "vehicle_type": r.vehicle_type,
        "direction": r.direction,
        "cargo_type": r.cargo_type,
        "load_status": r.load_status,
        "confidence_score": round(r.confidence_score or 0, 2),
        "plate_confidence": round(r.plate_confidence or 0, 2),
        "plate_status": r.plate_status or "PENDING",
        "plate_crop_url": r.plate_crop_url,
        "session_id": r.session_id,
        "event_time": _fmt(r.event_time),
        "snapshot_url": r.snapshot_url,
        "video_clip_url": r.video_clip_url,
        "status": r.status,
    }
