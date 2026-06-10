"""
安防 Demo 入口 — 无需专用硬件，视频文件即可

步骤:
  1. 把已知人员照片放入 data/face_db/姓名/*.jpg
  2. python -m security.enroll
  3. python -m security.run_security --video data/videos/xxx.mp4
     或不传 --video，自动下载示例视频

  笔记本摄像头: python -m security.run_security --camera
  保存输出视频: --save outputs/result.mp4
"""
import argparse
import os
import sys

from security.config import (
    ASYNC_CAMERA_DEFAULT,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DISPLAY_TARGET_FPS,
    FACE_DETECT_INTERVAL,
    MAX_FACE_CHECKS_PER_FRAME,
    OUTPUT_DIR,
    SERIAL_BAUD,
    SERIAL_CLOSE_DELAY_SEC,
    SERIAL_DEVICE,
    SERIAL_ENABLED_DEFAULT,
    YOLO_IMGSZ,
)
from security.face_database import FaceDatabase
from security.pipeline import ensure_sample_video, run_pipeline


def main():
    parser = argparse.ArgumentParser(description="安防: 检测+跟踪+人脸库+告警")
    parser.add_argument("--video", "-v", help="视频文件路径")
    parser.add_argument("--camera", "-c", action="store_true", help="使用 USB 摄像头")
    parser.add_argument("--device", type=int, default=0, help="摄像头索引，对应 /dev/videoN（默认 0）")
    parser.add_argument("--no-show", action="store_true", help="不显示窗口")
    parser.add_argument("--save", help="保存标注后的视频路径")
    parser.add_argument("--max-frames", type=int, help="最多处理帧数(测试用)")
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT, help="摄像头高度")
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ, help="YOLO 推理边长(越小越快)")
    parser.add_argument(
        "--face-interval", type=int, default=FACE_DETECT_INTERVAL,
        help="每 N 帧做人脸识别",
    )
    parser.add_argument(
        "--max-face", type=int, default=MAX_FACE_CHECKS_PER_FRAME,
        help="每轮最多识别几个行人",
    )
    parser.add_argument("--profile", action="store_true", help="画面显示 FPS（显示/推理分开统计）")
    parser.add_argument(
        "--sync", action="store_true",
        help="同步模式（推理完才显示，较慢）",
    )
    parser.add_argument(
        "--display-fps", type=int, default=DISPLAY_TARGET_FPS,
        help="窗口目标帧率（默认 30）",
    )
    parser.add_argument(
        "--serial-device", default=SERIAL_DEVICE,
        help="串口设备（auto 自动选择 ACM/USB）",
    )
    parser.add_argument("--serial-baud", type=int, default=SERIAL_BAUD)
    parser.add_argument("--no-serial", action="store_true", help="禁用串口门禁输出")
    parser.add_argument(
        "--serial-close-delay", type=float, default=SERIAL_CLOSE_DELAY_SEC,
        help="库内人员消失后延时发 close（秒）",
    )
    args = parser.parse_args()

    if args.camera:
        if sys.platform.startswith("linux"):
            os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_LIST", "V4L2")
        source = args.device
    elif args.video:
        source = args.video
    else:
        source = ensure_sample_video()
        print(f"使用示例视频: {source}")

    db = FaceDatabase()
    print(f"人脸库: {len(db.list_persons())} 人, {db.count()} 条特征")

    try:
        run_pipeline(
            source=source,
            db=db,
            show=not args.no_show,
            save_video=args.save,
            max_frames=args.max_frames,
            camera_width=args.width,
            camera_height=args.height,
            yolo_imgsz=args.imgsz,
            face_interval=args.face_interval,
            max_face_checks=args.max_face,
            profile=args.profile,
            async_display=args.camera and not args.no_show and not args.sync
            if ASYNC_CAMERA_DEFAULT
            else False,
            display_fps=args.display_fps,
            serial_device=(
                None if args.no_serial or not sys.platform.startswith("linux")
                else args.serial_device
            ),
            serial_baud=args.serial_baud,
            serial_close_delay_sec=args.serial_close_delay,
        )
    except KeyboardInterrupt:
        print("\n已中断")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        err = str(e).lower()
        if "remote" in err or "connection" in err or "download" in err:
            print(
                "\n多为首次下载模型失败（GitHub/Google 在板子上不稳定）。请先执行:\n"
                "  python3 -m security.download_models\n"
                "或在 PC 下载后 scp 到项目 models/20180402-114759-vggface2.pt",
                file=sys.stderr,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
