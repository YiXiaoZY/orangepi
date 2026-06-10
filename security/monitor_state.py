"""远程监控服务 — 全局状态（推理 + 香橙派心跳）"""
import time
from threading import Lock

_lock = Lock()
_started_at = time.time()
_last_infer_at: float | None = None
_last_infer_ms = 0.0
_last_infer_idx = 0
_last_overlays: list = []
_client: dict = {
    "online": False,
    "last_seen": None,
    "hostname": None,
    "camera_ok": False,
    "camera_resolution": None,
    "display_fps": 0.0,
    "infer_fps": 0.0,
    "serial_ok": False,
    "serial_state": "unknown",
    "serial_device": None,
    "serial_last_cmd": None,
}

CLIENT_TIMEOUT_SEC = 12.0


def record_infer(result: dict):
    global _last_infer_at, _last_infer_ms, _last_infer_idx, _last_overlays
    with _lock:
        _last_infer_at = time.time()
        _last_infer_ms = result.get("infer_ms", 0)
        _last_infer_idx = result.get("infer_idx", 0)
        _last_overlays = result.get("overlays", [])


def record_heartbeat(payload: dict):
    with _lock:
        _client.update({
            "online": True,
            "last_seen": time.time(),
            "hostname": payload.get("hostname"),
            "camera_ok": bool(payload.get("camera_ok")),
            "camera_resolution": payload.get("camera_resolution"),
            "display_fps": float(payload.get("display_fps") or 0),
            "infer_fps": float(payload.get("infer_fps") or 0),
            "serial_ok": bool(payload.get("serial_ok")),
            "serial_state": payload.get("serial_state", "unknown"),
            "serial_device": payload.get("serial_device"),
            "serial_last_cmd": payload.get("serial_last_cmd"),
        })


def get_snapshot() -> dict:
    now = time.time()
    with _lock:
        client = dict(_client)
        if client.get("last_seen"):
            client["online"] = (now - client["last_seen"]) < CLIENT_TIMEOUT_SEC
        else:
            client["online"] = False

        infer_online = (
            _last_infer_at is not None
            and (now - _last_infer_at) < CLIENT_TIMEOUT_SEC
        )

        known = [o for o in _last_overlays if o.get("label") != "陌生人"]
        strangers = [o for o in _last_overlays if o.get("label") == "陌生人"]

        return {
            "server": {
                "uptime_sec": round(now - _started_at),
                "engine_ok": True,
            },
            "inference": {
                "active": infer_online,
                "last_at": _last_infer_at,
                "last_ms": _last_infer_ms,
                "total_frames": _last_infer_idx,
                "persons": len(_last_overlays),
                "known": known,
                "strangers": strangers,
            },
            "client": client,
        }
