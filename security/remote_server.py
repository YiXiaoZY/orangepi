"""
PC 端推理服务 — 香橙派上传画面，本机 GPU/CPU 计算后返回框与姓名

  python -m security.remote_server --host 0.0.0.0 --port 8765

监控面板: http://<PC_IP>:8765/
"""
import argparse
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from security.inference_engine import SecurityInferenceEngine
from security.monitor_state import get_snapshot, record_heartbeat

_engine: SecurityInferenceEngine | None = None
_DASHBOARD = Path(__file__).parent / "dashboard.html"


class HeartbeatPayload(BaseModel):
    hostname: str | None = None
    camera_ok: bool = False
    camera_resolution: str | None = None
    display_fps: float = 0
    infer_fps: float = 0
    serial_ok: bool = False
    serial_state: str = "unknown"
    serial_device: str | None = None
    serial_last_cmd: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = SecurityInferenceEngine()
    yield
    if _engine:
        _engine.close()
        _engine = None


app = FastAPI(title="TE Security Remote Inference", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if not _DASHBOARD.is_file():
        return HTMLResponse("<h1>dashboard.html 缺失</h1>", status_code=500)
    return HTMLResponse(_DASHBOARD.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    if _engine is None:
        return JSONResponse({"ok": False}, status_code=503)
    return {
        "ok": True,
        "persons": len(_engine.db.list_persons()),
        "embeddings": _engine.db.count(),
        "infer_idx": _engine.infer_idx,
    }


@app.get("/api/status")
def api_status():
    if _engine is None:
        return JSONResponse({"ok": False}, status_code=503)
    data = get_snapshot()
    data["database"] = {
        "persons": len(_engine.db.list_persons()),
        "embeddings": _engine.db.count(),
    }
    return data


@app.post("/api/heartbeat")
def api_heartbeat(body: HeartbeatPayload):
    record_heartbeat(body.model_dump())
    return {"ok": True}


@app.post("/api/infer")
async def infer_frame(frame: UploadFile = File(...)):
    """multipart 字段名须为 frame（与 remote_client 一致）"""
    if _engine is None:
        return JSONResponse({"ok": False, "error": "引擎未就绪"}, status_code=503)
    data = await frame.read()
    if not data:
        return JSONResponse({"ok": False, "error": "空文件"}, status_code=400)
    return _engine.process_jpeg(data)


@app.post("/api/reset")
def reset_session():
    if _engine is None:
        return JSONResponse({"ok": False}, status_code=503)
    _engine.reset()
    return {"ok": True}


def main():
    parser = argparse.ArgumentParser(description="安防远程推理服务（跑在 PC）")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，0.0.0.0 允许局域网")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"监控面板: http://127.0.0.1:{args.port}/")
    print(f"API 状态: http://127.0.0.1:{args.port}/api/status")
    print("推理接口: POST /api/infer  (multipart 字段名 frame)")
    print(f"香橙派: python3 -m security.remote_client --server http://<PC_IP>:{args.port} --camera")
    uvicorn.run(
        "security.remote_server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
