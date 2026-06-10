"""
香橙派薄客户端 — 摄像头采集 + 窗口显示，推理在 PC 远程服务

  python3 -m security.remote_client --server http://192.168.x.x:8765 --camera --device 0
"""
import argparse
import socket
import sys
import threading
import time

import cv2
import numpy as np
import requests

from security.config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DISPLAY_TARGET_FPS,
    SERIAL_BAUD,
    SERIAL_CLOSE_CMD,
    SERIAL_CLOSE_DELAY_SEC,
    SERIAL_DEVICE,
    SERIAL_LINE_ENDING,
    SERIAL_MIN_OPEN_HOLD_SEC,
    SERIAL_OPEN_CMD,
    SERIAL_OPEN_STABLE_SEC,
)
from security.pipeline import _draw_overlays, _update_serial_gate, open_camera


class _OverlayStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.overlays = []
        self.infer_ms = 0.0
        self.infer_fps_ema = 0.0
        self.server_ok = False

    def update(self, overlays, infer_ms):
        with self._lock:
            self.overlays = overlays
            self.infer_ms = infer_ms
            self.server_ok = True

    def snapshot(self):
        with self._lock:
            return list(self.overlays), self.infer_ms, self.infer_fps_ema, self.server_ok


def _json_to_overlays(items):
    out = []
    for item in items:
        box = item["box"]
        color = tuple(item["color"])
        out.append((box, item["label"], color))
    return out


def _post_frame(server: str, jpeg_bytes: bytes, timeout: float) -> dict | None:
    url = server.rstrip("/") + "/api/infer"
    try:
        resp = requests.post(
            url,
            files={"frame": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        detail = ""
        if hasattr(e, "response") and e.response is not None:
            try:
                detail = e.response.text[:200]
            except Exception:
                pass
        print(f"[网络] 请求失败: {e} {detail}")
        return None


def run_remote_client(
    server: str,
    device: int,
    *,
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT,
    jpeg_quality=75,
    display_fps=DISPLAY_TARGET_FPS,
    profile=True,
    request_timeout=10.0,
    serial_device=SERIAL_DEVICE,
    serial_baud=SERIAL_BAUD,
    serial_close_delay_sec=SERIAL_CLOSE_DELAY_SEC,
    serial_open_stable_sec=SERIAL_OPEN_STABLE_SEC,
    serial_min_open_hold_sec=SERIAL_MIN_OPEN_HOLD_SEC,
):
    serial_gate = None
    if serial_device and sys.platform.startswith("linux"):
        from security.serial_gate import SerialGate

        serial_gate = SerialGate(
            device=serial_device,
            baud=serial_baud,
            line_ending=SERIAL_LINE_ENDING,
            open_cmd=SERIAL_OPEN_CMD,
            close_cmd=SERIAL_CLOSE_CMD,
            open_stable_sec=serial_open_stable_sec,
            close_delay_sec=serial_close_delay_sec,
            min_open_hold_sec=serial_min_open_hold_sec,
        )

    if sys.platform.startswith("linux"):
        import os
        os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_LIST", "V4L2")

    health_url = server.rstrip("/") + "/health"
    try:
        h = requests.get(health_url, timeout=5)
        h.raise_for_status()
        info = h.json()
        print(f"已连接 PC 服务: {server}")
        print(f"  人脸库: {info.get('persons')} 人, {info.get('embeddings')} 条特征")
    except requests.RequestException as e:
        print(f"无法连接 {health_url}: {e}")
        print("请先在 PC 启动: python -m security.remote_server")
        sys.exit(1)

    store = _OverlayStore()
    stop = threading.Event()
    latest_frame = None
    frame_lock = threading.Lock()
    live_stats = {"display_fps": 0.0}

    def capture_worker():
        nonlocal latest_frame
        cap = open_camera(device, width, height)
        try:
            while not stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue
                with frame_lock:
                    latest_frame = frame
        finally:
            cap.release()

    def network_worker():
        infer_t0 = time.perf_counter()
        infer_count = 0
        pending = {"bytes": None}
        pend_lock = threading.Lock()

        def offer_jpeg(jpeg_bytes):
            with pend_lock:
                pending["bytes"] = jpeg_bytes

        while not stop.is_set():
            with frame_lock:
                raw = latest_frame
            if raw is None:
                time.sleep(0.01)
                continue
            ok, enc = cv2.imencode(
                ".jpg", raw, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
            if ok:
                offer_jpeg(enc.tobytes())

            with pend_lock:
                jpeg = pending["bytes"]
            if jpeg is None:
                time.sleep(0.005)
                continue

            data = _post_frame(server, jpeg, request_timeout)
            if data and data.get("ok"):
                overlays = _json_to_overlays(data.get("overlays", []))
                store.update(overlays, data.get("infer_ms", 0))
                _update_serial_gate(serial_gate, overlays)
                infer_count += 1
                if profile and infer_count % 5 == 0:
                    elapsed = time.perf_counter() - infer_t0
                    if elapsed > 0:
                        inst = 5 / elapsed
                        with store._lock:
                            store.infer_fps_ema = (
                                inst if store.infer_fps_ema == 0
                                else 0.7 * store.infer_fps_ema + 0.3 * inst
                            )
                    infer_t0 = time.perf_counter()

    def heartbeat_worker():
        url = server.rstrip("/") + "/api/heartbeat"
        while not stop.is_set():
            with frame_lock:
                cam_ok = latest_frame is not None
            _, _, infer_fps, _ = store.snapshot()
            ser = (
                serial_gate.status_dict()
                if serial_gate
                else {"ok": False, "state": "disabled", "device": None, "last_cmd": None}
            )
            payload = {
                "hostname": socket.gethostname(),
                "camera_ok": cam_ok,
                "camera_resolution": f"{width}x{height}",
                "display_fps": live_stats["display_fps"],
                "infer_fps": infer_fps,
                "serial_ok": ser.get("ok", False),
                "serial_state": ser.get("state", "unknown"),
                "serial_device": ser.get("device"),
                "serial_last_cmd": ser.get("last_cmd"),
            }
            try:
                requests.post(url, json=payload, timeout=3)
            except requests.RequestException:
                pass
            time.sleep(3)

    cap_th = threading.Thread(target=capture_worker, daemon=True)
    net_th = threading.Thread(target=network_worker, daemon=True)
    hb_th = threading.Thread(target=heartbeat_worker, daemon=True)
    cap_th.start()
    net_th.start()
    hb_th.start()

    frame_period = 1.0 / max(display_fps, 1)
    next_tick = time.perf_counter()
    display_idx = 0
    disp_t0 = time.perf_counter()
    display_fps_ema = 0.0

    print(f"远程模式: 显示 {display_fps} FPS | 推理在 PC ({server})")

    try:
        while not stop.is_set():
            display_idx += 1
            with frame_lock:
                raw = latest_frame
            if raw is None:
                time.sleep(0.01)
                continue

            overlays, infer_ms, infer_fps, server_ok = store.snapshot()
            canvas = raw.copy()
            _draw_overlays(canvas, overlays)

            if profile:
                if display_idx % 15 == 0:
                    elapsed = time.perf_counter() - disp_t0
                    if elapsed > 0:
                        inst = 15 / elapsed
                        display_fps_ema = (
                            inst if display_fps_ema == 0
                            else 0.7 * display_fps_ema + 0.3 * inst
                        )
                        live_stats["display_fps"] = display_fps_ema
                    disp_t0 = time.perf_counter()
                status = "OK" if server_ok else "等待PC"
                cv2.putText(
                    canvas,
                    f"显示:{display_fps_ema:.0f} 远程推理:{infer_fps:.0f} {infer_ms:.0f}ms {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                )

            cv2.imshow("Security Remote", canvas)
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
        cap_th.join(timeout=2)
        net_th.join(timeout=2)
        if serial_gate is not None:
            serial_gate.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="香橙派远程推理客户端")
    parser.add_argument(
        "--server", "-s", required=True,
        help="PC 服务地址，如 http://192.168.1.100:8765",
    )
    parser.add_argument("--camera", "-c", action="store_true", help="USB 摄像头")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--display-fps", type=int, default=DISPLAY_TARGET_FPS)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--profile", action="store_true", default=True, help="显示 FPS（默认开）")
    parser.add_argument("--no-profile", action="store_true", help="不显示 FPS 叠加")
    parser.add_argument("--serial-device", default=SERIAL_DEVICE, help="串口设备")
    parser.add_argument("--serial-baud", type=int, default=SERIAL_BAUD)
    parser.add_argument("--no-serial", action="store_true", help="禁用串口门禁")
    parser.add_argument(
        "--serial-close-delay", type=float, default=SERIAL_CLOSE_DELAY_SEC,
        help="人离开后延时 close（秒）",
    )
    parser.add_argument(
        "--serial-open-stable", type=float, default=SERIAL_OPEN_STABLE_SEC,
        help="识别稳定多久才 open（秒，越小越快）",
    )
    parser.add_argument(
        "--serial-min-hold", type=float, default=SERIAL_MIN_OPEN_HOLD_SEC,
        help="open 后最少保持（秒）",
    )
    args = parser.parse_args()

    if not args.camera:
        parser.error("远程客户端需加 --camera")

    run_remote_client(
        args.server,
        args.device,
        width=args.width,
        height=args.height,
        display_fps=args.display_fps,
        jpeg_quality=args.jpeg_quality,
        request_timeout=args.timeout,
        profile=args.profile and not args.no_profile,
        serial_device=None if args.no_serial else args.serial_device,
        serial_baud=args.serial_baud,
        serial_close_delay_sec=args.serial_close_delay,
        serial_open_stable_sec=args.serial_open_stable,
        serial_min_open_hold_sec=args.serial_min_hold,
    )


if __name__ == "__main__":
    main()
