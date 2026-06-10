"""
第 7 课：人脸检测（OpenCV + Haar Cascade）

用法:
  python face_detect.py                    # 下载示例图并检测
  python face_detect.py --image 照片.jpg   # 检测指定图片
  python face_detect.py --camera           # 摄像头实时检测（按 q 退出）

Haar Cascade 是经典人脸检测方法：快速、无需 GPU、适合入门。
"""
import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(BASE_DIR, "images")
PICTURES_DIR = os.path.join(BASE_DIR, "pictures")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(PICTURES_DIR, exist_ok=True)

SAMPLE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
SAMPLE_PATH = os.path.join(IMAGES_DIR, "sample_lena.jpg")

# OpenCV 自带的 Haar 模型路径
CASCADE_PATH = os.path.join(
    cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
)


def load_detector():
    """加载 Haar 人脸检测器"""
    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError(f"未找到模型: {CASCADE_PATH}")
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("CascadeClassifier 加载失败")
    return detector


def ensure_sample_image():
    """确保有一张示例图可用于演示"""
    if os.path.exists(SAMPLE_PATH):
        return SAMPLE_PATH
    print(f"下载示例图: {SAMPLE_URL}")
    urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    return SAMPLE_PATH


def detect_faces(detector, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    在图像中检测人脸。

    参数:
        scale_factor: 图像金字塔缩放步长，越小越慢但越准
        min_neighbors: 候选框合并阈值，越大误检越少
        min_size: 忽略小于此尺寸的检测框

    返回:
        faces: list of (x, y, w, h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # 直方图均衡，改善光照

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces


def draw_faces(image, faces):
    """在原图上画绿色框和数量标注"""
    output = image.copy()
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"face {i + 1}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        f"count: {len(faces)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def detect_image(image_path, save_path=None, show=True):
    """检测单张图片"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    detector = load_detector()
    faces = detect_faces(detector, image)
    result = draw_faces(image, faces)

    print(f"图片: {image_path}")
    print(f"检测到 {len(faces)} 张人脸")
    for i, (x, y, w, h) in enumerate(faces):
        print(f"  face {i + 1}: x={x}, y={y}, w={w}, h={h}")

    if save_path:
        cv2.imwrite(save_path, result)
        print(f"结果已保存: {save_path}")

    if show:
        cv2.imshow("Face Detection", result)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return faces, result


def detect_camera(camera_id=0):
    """摄像头实时人脸检测，按 q 退出"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 (id={camera_id})。请检查设备或改用 --image"
        )

    detector = load_detector()
    print("摄像头已启动，按 q 退出")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取帧失败")
            break

        faces = detect_faces(detector, frame)
        frame = draw_faces(frame, faces)

        cv2.imshow("Face Detection - Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCV Haar 人脸检测")
    parser.add_argument("--image", "-i", help="待检测图片路径")
    parser.add_argument("--camera", "-c", action="store_true", help="使用摄像头")
    parser.add_argument(
        "--no-show", action="store_true", help="不弹窗，只保存结果"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_path = os.path.join(PICTURES_DIR, "face_detect_result.jpg")

    try:
        if args.camera:
            detect_camera()
        elif args.image:
            detect_image(args.image, save_path=save_path, show=not args.no_show)
        else:
            sample = ensure_sample_image()
            detect_image(sample, save_path=save_path, show=not args.no_show)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
