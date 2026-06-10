import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from security.model_assets import FACE_MODEL_PATH, ensure_blaze_face, load_facenet_model


class FaceEmbedder:
    """MediaPipe 裁脸 + FaceNet 512 维特征"""

    def __init__(self, device=None):
        ensure_blaze_face(verbose=False)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = load_facenet_model(self.device, verbose=True)
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)

    def close(self):
        self.face_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _detect_largest_face(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_detector.detect(mp_image)
        if not result.detections:
            return None
        h, w = bgr_image.shape[:2]
        best, best_area = None, 0
        for det in result.detections:
            b = det.bounding_box
            area = b.width * b.height
            if area > best_area:
                best_area = area
                x1 = max(0, int(b.origin_x))
                y1 = max(0, int(b.origin_y))
                x2 = min(w, x1 + int(b.width))
                y2 = min(h, y1 + int(b.height))
                best = bgr_image[y1:y2, x1:x2]
        return best

    def embed(self, bgr_image):
        """从 BGR 图像提取 L2 归一化特征，失败返回 None"""
        face = self._detect_largest_face(bgr_image)
        if face is None or face.size == 0:
            return None
        face = cv2.resize(face, (160, 160))
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5
        with torch.no_grad():
            emb = self.resnet(tensor.to(self.device)).cpu().numpy()[0]
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else None
