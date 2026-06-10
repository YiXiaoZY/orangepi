"""
USB 摄像头测试（香橙派 5 / Linux / Windows 通用）

用法:
  python test_usb_camera.py              # 自动扫描并测试
  python test_usb_camera.py --device 0   # 指定设备号
  python test_usb_camera.py --preview  # 预览 5 秒（需显示器或 VNC）
  python test_usb_camera.py --list-only # 只列出设备

香橙派上若打不开，先执行:
  ls -l /dev/video*
  groups $USER    # 确认在 video 组
  sudo usermod -aG video $USER  # 若无权限，加入后重新登录
"""
import argparse
import glob
import os
import re
import sys
import time

# 香橙派/Linux：优先 V4L2，避免 GStreamer 扫 /dev/video1 等非采集节点时报错
if sys.platform.startswith("linux"):
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_LIST", "V4L2")

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_probe_indices(max_index=8):
    """Linux 只探测存在的 /dev/videoN，减少无效警告"""
    if sys.platform.startswith("linux"):
        indices = []
        for path in sorted(glob.glob("/dev/video*")):
            m = re.search(r"video(\d+)$", path)
            if m:
                indices.append(int(m.group(1)))
        return sorted(set(indices)) or list(range(max_index))
    return list(range(max_index))


def list_video_devices(max_index=8):
    """尝试打开可用 index，返回可用设备列表"""
    found = []
    for i in get_probe_indices(max_index):
        cap = open_camera(i)
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()
            ok = ret and frame is not None
            found.append({
                "index": i,
                "ok": ok,
                "width": w,
                "height": h,
                "fps": fps,
                "backend": backend,
            })
            cap.release()
        elif cap is not None:
            cap.release()
    return found


def open_camera(device, width=None, height=None):
    """打开摄像头，Linux 上强制 V4L2"""
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        return None

    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def test_capture(device, num_frames=10, save_path=None):
    cap = open_camera(device)
    if cap is None:
        print(f"[失败] 无法打开设备 {device}")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_reported = cap.get(cv2.CAP_PROP_FPS)
    print(f"[打开] 设备 {device}: {w}x{h}, 报告 FPS={fps_reported}")

    ok_count = 0
    t0 = time.time()
    last_frame = None

    for i in range(num_frames):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok_count += 1
            last_frame = frame
        else:
            print(f"  第 {i + 1} 帧读取失败")

    elapsed = time.time() - t0
    real_fps = ok_count / elapsed if elapsed > 0 else 0
    print(f"[读帧] 成功 {ok_count}/{num_frames}, 耗时 {elapsed:.2f}s, 约 {real_fps:.1f} FPS")

    if last_frame is not None and save_path:
        cv2.imwrite(save_path, last_frame)
        print(f"[保存] 测试图: {save_path}")

    cap.release()
    success = ok_count == num_frames
    print("[通过]" if success else "[异常]", "摄像头采集" + ("正常" if success else "不稳定"))
    return success


def preview(device, seconds=5):
    cap = open_camera(device)
    if cap is None:
        print(f"[失败] 无法打开设备 {device}")
        return False

    print(f"预览设备 {device}，{seconds} 秒后自动结束，按 q 提前退出")
    end = time.time() + seconds
    while time.time() < end:
        ret, frame = cap.read()
        if not ret:
            print("读帧失败")
            break
        cv2.imshow("USB Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True


def print_linux_tips():
    if not sys.platform.startswith("linux"):
        return
    print("\n--- 香橙派 / Linux 排查提示 ---")
    print("  查看设备:  ls -l /dev/video*")
    print("  查看 USB:  lsusb")
    print("  权限不足:  sudo usermod -aG video $USER  然后重新登录")
    print("  占用检测:  fuser /dev/video0")
    print("  OpenCV:    python -c \"import cv2; print(cv2.__version__)\"")


def main():
    parser = argparse.ArgumentParser(description="USB 摄像头测试")
    parser.add_argument("--device", "-d", type=int, default=None, help="摄像头编号，默认自动选第一个可用")
    parser.add_argument("--list-only", action="store_true", help="只扫描设备")
    parser.add_argument("--preview", action="store_true", help="实时预览")
    parser.add_argument("--seconds", type=int, default=5, help="预览秒数")
    parser.add_argument("--frames", type=int, default=10, help="连续读帧测试数量")
    parser.add_argument("--width", type=int, default=None, help="尝试设置宽度")
    parser.add_argument("--height", type=int, default=None, help="尝试设置高度")
    args = parser.parse_args()

    print(f"平台: {sys.platform}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"输出目录: {OUTPUT_DIR}\n")

    devices = list_video_devices()
    if not devices:
        print("[失败] 未发现任何可用摄像头 (尝试了 index 0~7)")
        print_linux_tips()
        sys.exit(1)

    print("扫描到的设备:")
    for d in devices:
        status = "可读帧" if d["ok"] else "打开但读帧失败"
        print(
            f"  /dev/video{d['index']} (index={d['index']}): "
            f"{d['width']}x{d['height']} fps={d['fps']:.1f} "
            f"backend={d['backend']} [{status}]"
        )

    if args.list_only:
        print_linux_tips()
        return

    device = args.device
    if device is None:
        device = next((d["index"] for d in devices if d["ok"]), devices[0]["index"])
        print(f"\n自动选用设备 index={device}")

    if args.preview:
        ok = preview(device, args.seconds)
        sys.exit(0 if ok else 1)

    save_path = os.path.join(OUTPUT_DIR, f"usb_camera_test_{device}.jpg")
    if args.width and args.height:
        cap = open_camera(device, args.width, args.height)
        if cap:
            cap.release()
    ok = test_capture(device, args.frames, save_path)
    print_linux_tips()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
