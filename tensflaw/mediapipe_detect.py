"""
第 8 课：MediaPipe 人脸 + 手势检测

用法:
  python mediapipe_detect.py                      # 示例图，人脸检测
  python mediapipe_detect.py --mode face          # 仅人脸
  python mediapipe_detect.py --mode hand          # 手部关键点 + 手指数
  python mediapipe_detect.py --mode both          # 人脸 + 手
  python mediapipe_detect.py --image 照片.jpg
  python mediapipe_detect.py --camera             # 摄像头（按 q 退出）
"""
import argparse
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(BASE_DIR, "images")
PICTURES_DIR = os.path.join(BASE_DIR, "pictures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(PICTURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SAMPLE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
SAMPLE_PATH = os.path.join(IMAGES_DIR, "sample_lena.jpg")

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

FINGER_TIPS = [4, 8, 12, 16, 20]


def download_if_missing(url, path):
    if os.path.exists(path):
        return path
    print(f"下载: {url}")
    urllib.request.urlretrieve(url, path)
    return path


def ensure_sample_image():
    if os.path.exists(SAMPLE_PATH):
        return SAMPLE_PATH
    print(f"下载示例图: {SAMPLE_URL}")
    urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    return SAMPLE_PATH


def bgr_to_mp_image(bgr_image):
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def create_face_detector(running_mode=vision.RunningMode.IMAGE):
    download_if_missing(FACE_MODEL_URL, FACE_MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=running_mode,
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(options)


def create_hand_landmarker(running_mode=vision.RunningMode.IMAGE):
    download_if_missing(HAND_MODEL_URL, HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_faces(image, detection_result):
    for det in detection_result.detections:
        bbox = det.bounding_box
        x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
        x2, y2 = x1 + int(bbox.width), y1 + int(bbox.height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        score = det.categories[0].score if det.categories else 0.0
        cv2.putText(
            image,
            f"face {score:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        f"faces: {len(detection_result.detections)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def get_hand_label(handedness_list):
    """MediaPipe 0.10+: handedness[i] 是 Category 列表，不是 .classification"""
    if not handedness_list:
        return "Unknown"
    return handedness_list[0].category_name or "Unknown"


def count_fingers(hand_landmarks, handedness_list):
    lm = hand_landmarks
    fingers = 0
    is_left = get_hand_label(handedness_list) == "Left"

    if is_left:
        if lm[FINGER_TIPS[0]].x < lm[FINGER_TIPS[0] - 1].x:
            fingers += 1
    else:
        if lm[FINGER_TIPS[0]].x > lm[FINGER_TIPS[0] - 1].x:
            fingers += 1

    for tip_id in FINGER_TIPS[1:]:
        if lm[tip_id].y < lm[tip_id - 2].y:
            fingers += 1
    return fingers


def draw_hands(image, hand_result):
    h, w = image.shape[:2]
    if not hand_result.hand_landmarks:
        cv2.putText(
            image, "hands: 0", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA,
        )
        return image

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
        for a, b in connections:
            pa, pb = hand_landmarks[a], hand_landmarks[b]
            cv2.line(
                image,
                (int(pa.x * w), int(pa.y * h)),
                (int(pb.x * w), int(pb.y * h)),
                (0, 165, 255),
                2,
            )
        for lm in hand_landmarks:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 3, (255, 0, 0), -1)

        if i < len(hand_result.handedness):
            handedness_list = hand_result.handedness[i]
            label = get_hand_label(handedness_list)
            fingers = count_fingers(hand_landmarks, handedness_list)
            wrist = hand_landmarks[0]
            cv2.putText(
                image,
                f"{label} {fingers} fingers",
                (int(wrist.x * w), int(wrist.y * h) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        image,
        f"hands: {len(hand_result.hand_landmarks)}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return image


def run_detectors(frame, mode, face_detector, hand_landmarker, timestamp_ms=0, is_video=False):
    output = frame.copy()
    mp_image = bgr_to_mp_image(frame)

    if face_detector is not None:
        if is_video:
            face_result = face_detector.detect_for_video(mp_image, timestamp_ms)
        else:
            face_result = face_detector.detect(mp_image)
        output = draw_faces(output, face_result)

    if hand_landmarker is not None:
        if is_video:
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            hand_result = hand_landmarker.detect(mp_image)
        output = draw_hands(output, hand_result)

    return output


def detect_image(image_path, mode, save_path=None, show=True):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"无法读取: {image_path}")

    use_face = mode in ("face", "both")
    use_hand = mode in ("hand", "both")

    with create_face_detector() if use_face else _noop_context() as face_det, \
         create_hand_landmarker() if use_hand else _noop_context() as hand_det:
        output = run_detectors(
            frame, mode,
            face_det if use_face else None,
            hand_det if use_hand else None,
            is_video=False,
        )

    print(f"图片: {image_path}  模式: {mode}")
    if save_path:
        cv2.imwrite(save_path, output)
        print(f"结果已保存: {save_path}")

    if show:
        cv2.imshow("MediaPipe", output)
        print("按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output


def detect_camera(mode, camera_id=0):
    use_face = mode in ("face", "both")
    use_hand = mode in ("hand", "both")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 id={camera_id}")

    video_mode = vision.RunningMode.VIDEO
    with create_face_detector(video_mode) if use_face else _noop_context() as face_det, \
         create_hand_landmarker(video_mode) if use_hand else _noop_context() as hand_det:
        print("摄像头已启动，按 q 退出")
        start = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int((time.time() - start) * 1000)
            output = run_detectors(
                frame, mode,
                face_det if use_face else None,
                hand_det if use_hand else None,
                timestamp_ms,
                is_video=True,
            )
            cv2.imshow("MediaPipe - Camera", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


class _noop_context:
    """占位 context manager，mode 不含 face/hand 时使用"""
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="MediaPipe 人脸/手势检测")
    parser.add_argument("--image", "-i", help="图片路径")
    parser.add_argument("--camera", "-c", action="store_true", help="摄像头")
    parser.add_argument(
        "--mode", "-m",
        choices=["face", "hand", "both"],
        default="face",
        help="face=人脸, hand=手部+手指数, both=两者",
    )
    parser.add_argument("--no-show", action="store_true", help="不弹窗")
    return parser.parse_args()


def main():
    args = parse_args()
    save_path = os.path.join(PICTURES_DIR, f"mediapipe_{args.mode}.jpg")

    try:
        if args.camera:
            detect_camera(args.mode)
        elif args.image:
            detect_image(args.image, args.mode, save_path, show=not args.no_show)
        else:
            sample = ensure_sample_image()
            detect_image(sample, args.mode, save_path, show=not args.no_show)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
