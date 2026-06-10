import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from security.alert_manager import AlertManager
from security.config import (
    ALERT_DIR,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    FACE_DETECT_INTERVAL,
    MAX_FACE_CHECKS_PER_FRAME,
    MATCH_THRESHOLD,
    PERSON_CLASS_ID,
    SAMPLE_VIDEO_PATH,
    SAMPLE_VIDEO_URL,
    DISPLAY_TARGET_FPS,
    TORCH_NUM_THREADS,
    TRACKER,
    YOLO_CONF,
    YOLO_IMGSZ,
)
from security.face_database import FaceDatabase
from security.face_embedder import FaceEmbedder
from security.model_assets import check_models_ready, ensure_yolo, get_yolo_model_path


def ensure_sample_video():
    import os
    if os.path.exists(SAMPLE_VIDEO_PATH):
        return SAMPLE_VIDEO_PATH
    os.makedirs(os.path.dirname(SAMPLE_VIDEO_PATH), exist_ok=True)
    print(f"下载示例视频: {SAMPLE_VIDEO_URL}")
    urllib.request.urlretrieve(SAMPLE_VIDEO_URL, SAMPLE_VIDEO_PATH)
    return SAMPLE_VIDEO_PATH


def open_camera(device: int, width: int, height: int):
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 device={device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头: {actual_w}x{actual_h}")
    return cap


def crop_person_upper_body(frame, box, upper_ratio=0.65):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    ph = y2 - y1
    y2 = min(h, y1 + int(ph * upper_ratio))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _parse_tracks(result):
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    tracks = list(
        zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.int().cpu().tolist(),
        )
    )
    tracks.sort(key=lambda t: _box_area(t[0]), reverse=True)
    return tracks


def _update_identity(
    frame,
    tracks,
    infer_idx,
    embedder,
    db,
    alert_mgr,
    identity_cache,
    face_interval,
    max_face_checks,
):
    if infer_idx % face_interval != 0:
        return
    for box, track_id in tracks[:max_face_checks]:
        tid = int(track_id)
        if tid in identity_cache and not identity_cache[tid]["is_stranger"]:
            continue
        crop = crop_person_upper_body(frame, box)
        if crop is None:
            continue
        emb = embedder.embed(crop)
        if emb is None:
            continue
        name, score = db.match(emb)
        if name:
            identity_cache[tid] = {"name": name, "is_stranger": False}
        else:
            identity_cache[tid] = {"name": None, "is_stranger": True}
            if alert_mgr.should_alert(tid):
                snap = alert_mgr.save_snapshot(frame, tid, "stranger")
                db.log_alert(tid, "stranger", None, score, snap)
                print(f"[告警] 陌生人 -> {snap}")


def _build_overlays(tracks, identity_cache):
    """返回 [(box, label, color), ...] 供显示线程绘制"""
    overlays = []
    for box, track_id in tracks:
        tid = int(track_id)
        info = identity_cache.get(tid)
        if info is None:
            continue
        if info["is_stranger"]:
            overlays.append((box, "陌生人", (0, 0, 255)))
        else:
            overlays.append((box, info["name"], (0, 255, 0)))
    return overlays


def _has_known_person(overlays) -> bool:
    return any(label != "陌生人" for _, label, _ in overlays)


def _update_serial_gate(serial_gate, overlays):
    if serial_gate is not None:
        serial_gate.update_known_presence(_has_known_person(overlays))


def _draw_overlays(frame, overlays):
    for box, label, color in overlays:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )


def _process_frame(
    frame,
    result,
    frame_idx,
    embedder,
    db,
    alert_mgr,
    identity_cache,
    face_interval,
    max_face_checks,
    serial_gate=None,
):
    tracks = _parse_tracks(result)
    _update_identity(
        frame, tracks, frame_idx, embedder, db, alert_mgr,
        identity_cache, face_interval, max_face_checks,
    )
    overlays = _build_overlays(tracks, identity_cache)
    _update_serial_gate(serial_gate, overlays)
    _draw_overlays(frame, overlays)


class _LatestFrame:
    """推理线程只处理最新一帧，避免积压"""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._event = threading.Event()

    def offer(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame
        self._event.set()

    def take(self, stop: threading.Event) -> np.ndarray | None:
        while not stop.is_set():
            if self._event.wait(timeout=0.05):
                with self._lock:
                    frame = self._frame
                    self._frame = None
                    self._event.clear()
                if frame is not None:
                    return frame
        return None


def _run_camera_async(
    source: int,
    yolo,
    track_kw,
    embedder,
    db,
    alert_mgr,
    *,
    camera_width,
    camera_height,
    face_interval,
    max_face_checks,
    profile,
    display_fps,
    max_frames,
    serial_gate=None,
):
    broker = _LatestFrame()
    stop = threading.Event()
    identity_cache = {}
    overlays = []
    display_fps_ema = 0.0
    infer_fps_ema = 0.0
    lock = threading.Lock()
    infer_idx = 0

    def worker():
        nonlocal infer_idx, infer_fps_ema
        infer_t0 = time.perf_counter()
        infer_count = 0
        while not stop.is_set():
            frame = broker.take(stop)
            if frame is None:
                continue
            infer_idx += 1
            result = yolo.track(frame, **track_kw)[0]
            tracks = _parse_tracks(result)
            _update_identity(
                frame, tracks, infer_idx, embedder, db, alert_mgr,
                identity_cache, face_interval, max_face_checks,
            )
            new_overlays = _build_overlays(tracks, identity_cache)
            with lock:
                overlays.clear()
                overlays.extend(new_overlays)
            _update_serial_gate(serial_gate, new_overlays)
            infer_count += 1
            if profile and infer_count % 10 == 0:
                elapsed = time.perf_counter() - infer_t0
                if elapsed > 0:
                    inst = 10 / elapsed
                    infer_fps_ema = inst if infer_fps_ema == 0 else 0.7 * infer_fps_ema + 0.3 * inst
                infer_t0 = time.perf_counter()

    latest_frame = None

    def capture_worker():
        nonlocal latest_frame
        cap = open_camera(source, camera_width, camera_height)
        try:
            while not stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue
                with lock:
                    latest_frame = frame
                broker.offer(frame)
        finally:
            cap.release()

    th = threading.Thread(target=worker, daemon=True)
    cap_th = threading.Thread(target=capture_worker, daemon=True)
    th.start()
    cap_th.start()

    frame_period = 1.0 / max(display_fps, 1)
    next_tick = time.perf_counter()
    display_idx = 0
    disp_t0 = time.perf_counter()

    print(
        f"异步模式: 显示目标 {display_fps} FPS | "
        f"推理 imgsz={track_kw['imgsz']} | 人脸每 {face_interval} 次推理"
    )

    try:
        while not stop.is_set():
            display_idx += 1
            if max_frames and display_idx > max_frames:
                break

            with lock:
                raw = latest_frame
                current = list(overlays)
            if raw is None:
                time.sleep(0.01)
                continue

            canvas = raw.copy()
            _draw_overlays(canvas, current)

            if profile:
                if display_idx % 15 == 0:
                    elapsed = time.perf_counter() - disp_t0
                    if elapsed > 0:
                        inst = 15 / elapsed
                        display_fps_ema = (
                            inst if display_fps_ema == 0
                            else 0.7 * display_fps_ema + 0.3 * inst
                        )
                    disp_t0 = time.perf_counter()
                cv2.putText(
                    canvas,
                    f"显示:{display_fps_ema:.0f} 推理:{infer_fps_ema:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

            cv2.imshow("Security Pipeline", canvas)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            next_tick += frame_period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()
    finally:
        stop.set()
        th.join(timeout=2.0)
        cap_th.join(timeout=2.0)


def run_pipeline(
    source,
    db=None,
    show=True,
    save_video=None,
    max_frames=None,
    *,
    camera_width=CAMERA_WIDTH,
    camera_height=CAMERA_HEIGHT,
    yolo_imgsz=YOLO_IMGSZ,
    yolo_conf=YOLO_CONF,
    face_interval=FACE_DETECT_INTERVAL,
    max_face_checks=MAX_FACE_CHECKS_PER_FRAME,
    profile=False,
    async_display=False,
    display_fps=DISPLAY_TARGET_FPS,
    serial_device="auto",
    serial_baud=115200,
    serial_close_delay_sec=1.0,
):
    """
    安防主流程: YOLO 行人跟踪 + 人脸库比对 + 陌生人告警

    source: 视频路径、摄像头 id(0)、或 RTSP url
    """
    torch.set_num_threads(TORCH_NUM_THREADS)

    db = db or FaceDatabase()
    if db.count() == 0:
        print("警告: 人脸库为空，请先运行 enroll.py 注册人员")

    missing = check_models_ready()
    if missing:
        print("本地模型未齐，尝试下载（也可先: python3 -m security.download_models）")
        ensure_yolo(verbose=True)

    yolo_weights = get_yolo_model_path()
    print(f"加载 YOLO: {yolo_weights} (imgsz={yolo_imgsz})")
    yolo = YOLO(yolo_weights)

    track_kw = dict(
        persist=True,
        classes=[PERSON_CLASS_ID],
        tracker=TRACKER,
        verbose=False,
        imgsz=yolo_imgsz,
        conf=yolo_conf,
        device="cpu",
    )

    alert_mgr = AlertManager()
    serial_gate = None
    if serial_device and sys.platform.startswith("linux"):
        from security.config import (
            SERIAL_CLOSE_CMD,
            SERIAL_LINE_ENDING,
            SERIAL_MIN_OPEN_HOLD_SEC,
            SERIAL_OPEN_CMD,
            SERIAL_OPEN_STABLE_SEC,
        )
        from security.serial_gate import SerialGate

        serial_gate = SerialGate(
            device=serial_device,
            baud=serial_baud,
            line_ending=SERIAL_LINE_ENDING,
            open_cmd=SERIAL_OPEN_CMD,
            close_cmd=SERIAL_CLOSE_CMD,
            open_stable_sec=SERIAL_OPEN_STABLE_SEC,
            close_delay_sec=serial_close_delay_sec,
            min_open_hold_sec=SERIAL_MIN_OPEN_HOLD_SEC,
        )

    frame_idx = 0
    identity_cache = {}
    writer = None
    t0 = time.perf_counter()
    fps_ema = None

    print("加载人脸模型...")
    with FaceEmbedder() as embedder:
        print(
            f"启动检测 (人脸每 {face_interval} 帧, 每帧最多 {max_face_checks} 人, "
            f"阈值 {MATCH_THRESHOLD})..."
        )

        if isinstance(source, int) and show and async_display:
            _run_camera_async(
                source, yolo, track_kw, embedder, db, alert_mgr,
                camera_width=camera_width,
                camera_height=camera_height,
                face_interval=face_interval,
                max_face_checks=max_face_checks,
                profile=profile,
                display_fps=display_fps,
                max_frames=max_frames,
                serial_gate=serial_gate,
            )
        elif isinstance(source, int):
            cap = open_camera(source, camera_width, camera_height)
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    if max_frames and frame_idx > max_frames:
                        break

                    result = yolo.track(frame, **track_kw)[0]
                    _process_frame(
                        frame, result, frame_idx, embedder, db, alert_mgr,
                        identity_cache, face_interval, max_face_checks,
                        serial_gate=serial_gate,
                    )

                    if writer:
                        writer.write(frame)
                    if show:
                        if profile and frame_idx % 30 == 0:
                            elapsed = time.perf_counter() - t0
                            inst_fps = 30 / elapsed if elapsed > 0 else 0
                            fps_ema = inst_fps if fps_ema is None else 0.7 * fps_ema + 0.3 * inst_fps
                            t0 = time.perf_counter()
                            cv2.putText(
                                frame, f"FPS:{fps_ema:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            )
                        cv2.imshow("Security Pipeline", frame)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            break
            finally:
                cap.release()
        else:
            if save_video:
                cap_probe = cv2.VideoCapture(source)
                w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25
                cap_probe.release()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_video, fourcc, fps, (w, h))

            for result in yolo.track(source=source, stream=True, **track_kw):
                frame = result.orig_img
                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break
                _process_frame(
                    frame, result, frame_idx, embedder, db, alert_mgr,
                    identity_cache, face_interval, max_face_checks,
                    serial_gate=serial_gate,
                )
                if writer:
                    writer.write(frame)
                if show:
                    cv2.imshow("Security Pipeline", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

    if serial_gate is not None:
        serial_gate.close()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"处理结束，告警截图目录: {ALERT_DIR}")
