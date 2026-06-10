"""PC 端推理引擎 — 供 remote_server 与本地 pipeline 共用逻辑"""
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from security.alert_manager import AlertManager
from security.config import (
    FACE_DETECT_INTERVAL,
    MAX_FACE_CHECKS_PER_FRAME,
    TORCH_NUM_THREADS,
    TRACKER,
    YOLO_CONF,
    YOLO_IMGSZ,
    PERSON_CLASS_ID,
)
from security.face_database import FaceDatabase
from security.face_embedder import FaceEmbedder
from security.model_assets import ensure_yolo, get_yolo_model_path
from security.pipeline import (
    _build_overlays,
    _parse_tracks,
    _update_identity,
)


def _overlay_to_json(overlays):
    out = []
    for box, label, color in overlays:
        out.append({
            "box": [float(x) for x in box],
            "label": label,
            "color": [int(c) for c in color],
        })
    return out


class SecurityInferenceEngine:
    """单路摄像头会话：YOLO 跟踪 + 人脸库 + 告警"""

    def __init__(
        self,
        *,
        yolo_imgsz=YOLO_IMGSZ,
        yolo_conf=YOLO_CONF,
        face_interval=FACE_DETECT_INTERVAL,
        max_face_checks=MAX_FACE_CHECKS_PER_FRAME,
        db=None,
    ):
        torch.set_num_threads(TORCH_NUM_THREADS)
        ensure_yolo(verbose=True)
        weights = get_yolo_model_path()
        print(f"[推理引擎] YOLO: {weights} imgsz={yolo_imgsz}")
        self.yolo = YOLO(weights)
        self.track_kw = dict(
            persist=True,
            classes=[PERSON_CLASS_ID],
            tracker=TRACKER,
            verbose=False,
            imgsz=yolo_imgsz,
            conf=yolo_conf,
            device="cpu",
        )
        self.face_interval = face_interval
        self.max_face_checks = max_face_checks
        self.db = db or FaceDatabase()
        self.alert_mgr = AlertManager()
        self.identity_cache = {}
        self.infer_idx = 0
        self.embedder = FaceEmbedder()
        print(
            f"[推理引擎] 人脸库 {len(self.db.list_persons())} 人, "
            f"{self.db.count()} 条特征"
        )

    def close(self):
        self.embedder.close()

    def process_jpeg(self, jpeg_bytes: bytes) -> dict:
        t0 = time.perf_counter()
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "error": "无法解码 JPEG", "overlays": []}

        self.infer_idx += 1
        messages = []
        result = self.yolo.track(frame, **self.track_kw)[0]
        tracks = _parse_tracks(result)

        _update_identity(
            frame, tracks, self.infer_idx, self.embedder, self.db,
            self.alert_mgr, self.identity_cache, self.face_interval,
            self.max_face_checks,
        )

        overlays = _overlay_to_json(_build_overlays(tracks, self.identity_cache))
        infer_ms = (time.perf_counter() - t0) * 1000
        result = {
            "ok": True,
            "overlays": overlays,
            "infer_ms": round(infer_ms, 1),
            "infer_idx": self.infer_idx,
        }
        from security.monitor_state import record_infer
        record_infer(result)
        return result

    def reset(self):
        self.identity_cache.clear()
        self.infer_idx = 0
        self.yolo = YOLO(get_yolo_model_path())
