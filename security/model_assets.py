"""模型文件检查与下载（香橙派网络不稳时需先单独执行 download_models）"""
import os
import time
import urllib.request

from security.config import BASE_DIR, MODELS_DIR

YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
YOLO_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
)

FACE_MODEL_PATH = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

FACENET_RELEASE_URL = (
    "https://github.com/timesler/facenet-pytorch/releases/download/"
    "v2.2.9/20180402-114759-vggface2.pt"
)
FACENET_LOCAL_CANDIDATES = (
    os.path.join(MODELS_DIR, "20180402-114759-vggface2.pt"),
    os.path.join(MODELS_DIR, "vggface2.pt"),
)

YOLO_FALLBACK_PATH = os.path.join(BASE_DIR, "yolov8n.pt")


def resolve_yolo_path() -> str | None:
    """models/yolov8n.pt 或项目根目录 yolov8n.pt"""
    for path in (YOLO_MODEL_PATH, YOLO_FALLBACK_PATH):
        if os.path.isfile(path) and os.path.getsize(path) >= 1_000_000:
            return path
    return None


def get_yolo_model_path() -> str:
    return resolve_yolo_path() or YOLO_MODEL_PATH


def _download(url: str, dest: str, retries: int = 5, min_bytes: int = 1000) -> str:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        print(f"  已有: {dest} ({os.path.getsize(dest)} bytes)")
        return dest

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            print(f"  下载 ({attempt}/{retries}): {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            if os.path.getsize(dest) < min_bytes:
                raise OSError(f"文件过小: {dest}")
            print(f"  完成: {dest}")
            return dest
        except Exception as e:
            last_err = e
            print(f"  失败: {e}")
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"无法下载 {url} -> {dest}: {last_err}") from last_err


def ensure_yolo(verbose: bool = True) -> str:
    if verbose:
        print("[1/3] YOLO yolov8n.pt")
    existing = resolve_yolo_path()
    if existing:
        if verbose:
            print(f"  已有: {existing}")
        return existing
    return _download(YOLO_URL, YOLO_MODEL_PATH, min_bytes=1_000_000)


def ensure_blaze_face(verbose: bool = True) -> str:
    if verbose:
        print("[2/3] MediaPipe blaze_face")
    if os.path.isfile(FACE_MODEL_PATH) and os.path.getsize(FACE_MODEL_PATH) >= 10_000:
        if verbose:
            print(f"  已有: {FACE_MODEL_PATH}")
        return FACE_MODEL_PATH
    return _download(FACE_MODEL_URL, FACE_MODEL_PATH, min_bytes=10_000)


def resolve_facenet_weights() -> str | None:
    """优先 models/ 下的单文件权重（约 107MB）"""
    for path in FACENET_LOCAL_CANDIDATES:
        if os.path.isfile(path) and os.path.getsize(path) >= 1_000_000:
            return path
    ckpt = _torch_checkpoints_dir()
    if os.path.isdir(ckpt):
        for name in os.listdir(ckpt):
            if name.endswith(".pt") and (
                "vggface2" in name or "20180402-114759" in name
            ):
                path = os.path.join(ckpt, name)
                if os.path.getsize(path) >= 1_000_000:
                    return path
    return None


def _facenet_split_cache_ready() -> bool:
    """facenet-pytorch 分片缓存（两个 vggface2_*.pt）"""
    ckpt = _torch_checkpoints_dir()
    if not os.path.isdir(ckpt):
        return False
    parts = [
        n for n in os.listdir(ckpt)
        if n.endswith(".pt") and n.startswith("vggface2_")
    ]
    return len(parts) >= 2


def load_facenet_model(device: str = "cpu", verbose: bool = False):
    """加载 FaceNet；无本地权重时不联网，避免香橙派超时"""
    import torch
    from facenet_pytorch import InceptionResnetV1

    local = resolve_facenet_weights()
    if local:
        if verbose:
            print(f"  FaceNet 本地: {local}")
        # classify=False 输出 512 维特征；官方 .pt 含 logits 需剔除
        model = InceptionResnetV1(pretrained=None, classify=False).eval()
        try:
            state = torch.load(local, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(local, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {
            k: v for k, v in state.items()
            if not k.startswith("logits.")
        }
        model.load_state_dict(state, strict=True)
        return model.to(device)

    if _facenet_split_cache_ready():
        if verbose:
            print(f"  FaceNet 缓存: {_torch_checkpoints_dir()}")
        return InceptionResnetV1(pretrained="vggface2").eval().to(device)

    raise RuntimeError(
        "未找到 FaceNet 权重（不会自动联网下载）。请任选一种方式:\n"
        f"  1) 下载放到 models/20180402-114759-vggface2.pt\n"
        f"     {FACENET_RELEASE_URL}\n"
        f"  2) PC 执行 python -m security.download_models 后 scp:\n"
        f"     models/20180402-114759-vggface2.pt\n"
        f"     或整个 ~/.cache/torch/checkpoints/ 到板子同路径"
    )


def ensure_facenet(verbose: bool = True) -> None:
    if verbose:
        print("[3/3] FaceNet vggface2")
    load_facenet_model(device="cpu", verbose=verbose)
    if verbose:
        print("  FaceNet 权重已就绪")


def _torch_checkpoints_dir() -> str:
    try:
        from facenet_pytorch.models.inception_resnet_v1 import get_torch_home

        return os.path.join(get_torch_home(), "checkpoints")
    except Exception:
        return os.path.expanduser("~/.cache/torch/checkpoints")


def ensure_all_models(verbose: bool = True, facenet: bool = True) -> None:
    ensure_yolo(verbose=verbose)
    ensure_blaze_face(verbose=verbose)
    if facenet:
        ensure_facenet(verbose=verbose)


def check_models_ready() -> list[str]:
    """返回缺失项说明列表，空表示本地文件齐全（不含 FaceNet 缓存探测）"""
    missing = []
    if not resolve_yolo_path():
        missing.append(
            f"缺少 YOLO: {YOLO_MODEL_PATH} 或 {YOLO_FALLBACK_PATH}"
        )
    if not os.path.isfile(FACE_MODEL_PATH):
        missing.append(f"缺少 {FACE_MODEL_PATH}")
    return missing
