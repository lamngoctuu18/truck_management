"""Event bus: cầu nối giữa pipeline (thread nền) và các WebSocket client.

Pipeline gọi publish() từ thread xử lý video (đồng bộ). Bus đẩy vào asyncio.Queue
của từng client thông qua loop chính.
"""
import asyncio
from app.core.logger import get_logger

log = get_logger("EventBus")


class EventBus:
    def __init__(self):
        self._clients = set()          # set[asyncio.Queue]
        self._loop = None

    def bind_loop(self, loop):
        self._loop = loop

    def register(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=100)
        self._clients.add(q)
        return q

    def unregister(self, q):
        self._clients.discard(q)

    def publish(self, payload: dict):
        """Gọi từ thread nền (không phải async)."""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._dispatch, payload)
        except RuntimeError:
            pass

    def _dispatch(self, payload):
        for q in list(self._clients):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # bỏ client chậm 1 message thay vì block
                pass
