"""
OpenCV 调用 USB 摄像头（香橙派 5 / Linux 优先 V4L2）

用法:
  python usb_camera.py                    # 预览，按 q 退出
  python usb_camera.py --device 0         # 指定 /dev/video0
  python usb_camera.py --width 640 --height 480
  python usb_camera.py --snapshot out.jpg # 拍一张保存
  python usb_camera.py --record out.mp4   # 录像，按 q 结束
"""
import argparse
import os
import sys
import time

if sys.platform.startswith("linux"):
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_LIST", "V4L2")

import cv2

DEFAULT_DEVICE = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


def open_usb_camera(device, width=None, height=None):
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


def print_camera_info(cap, device):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"设备 index={device}  分辨率={w}x{h}  FPS={fps:.1f}  后端={cap.getBackendName()}")


def run_preview(cap, window_name="USB Camera"):
    print("预览中，按 q 退出")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("读帧失败")
            break
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def run_snapshot(cap, path):
    ret, frame = cap.read()
    if not ret or frame is None:
        print("拍照失败")
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, frame)
    print(f"已保存: {path}")
    return True


def run_record(cap, path, device):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print("无法创建视频文件，可尝试 .avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        path = path.rsplit(".", 1)[0] + ".avi"
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    print(f"录像中 -> {path}，按 q 结束")
    frames = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("读帧失败，停止录像")
            break
        writer.write(frame)
        frames += 1
        cv2.imshow("USB Camera (recording)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print(f"已保存 {frames} 帧, 时长 {elapsed:.1f}s, 约 {frames / elapsed:.1f} FPS")
    return True


def main():
    parser = argparse.ArgumentParser(description="OpenCV USB 摄像头")
    parser.add_argument("--device", "-d", type=int, default=DEFAULT_DEVICE)
    parser.add_argument("--width", "-W", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", "-H", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--snapshot", "-s", metavar="PATH", help="拍一张图并退出")
    parser.add_argument("--record", "-r", metavar="PATH", help="录像保存路径")
    parser.add_argument("--no-show", action="store_true", help="snapshot 时不弹窗")
    args = parser.parse_args()

    cap = open_usb_camera(args.device, args.width, args.height)
    if cap is None:
        print(f"无法打开摄像头 device={args.device}")
        print("请检查: ls -l /dev/video*  权限: groups | grep video")
        sys.exit(1)

    print_camera_info(cap, args.device)

    try:
        if args.snapshot:
            ok = run_snapshot(cap, args.snapshot)
            sys.exit(0 if ok else 1)
        if args.record:
            ok = run_record(cap, args.record, args.device)
            sys.exit(0 if ok else 1)
        if args.no_show and not args.snapshot:
            print("无显示模式请配合 --snapshot 或 --record")
            sys.exit(1)
        run_preview(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
