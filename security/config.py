import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SECURITY_DIR = os.path.join(BASE_DIR, "security")

DATA_DIR = os.path.join(BASE_DIR, "data")
FACE_DB_DIR = os.path.join(DATA_DIR, "face_db")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ALERT_DIR = os.path.join(OUTPUT_DIR, "alerts")

DB_PATH = os.path.join(DATA_DIR, "security.db")

MODELS_DIR = os.path.join(BASE_DIR, "models")

# YOLO 行人检测 + ByteTrack（优先使用项目 models/ 目录）
YOLO_MODEL = os.path.join(MODELS_DIR, "yolov8n.pt")
TRACKER = "bytetrack.yaml"
PERSON_CLASS_ID = 0

# 人脸识别
MATCH_THRESHOLD = 0.65          # 余弦相似度，低于此视为陌生人
ALERT_COOLDOWN_SEC = 30         # 同一 track_id 告警冷却
FACE_DETECT_INTERVAL = 10       # 每 N 帧做一次人脸比对（香橙派建议 10~20）
MAX_FACE_CHECKS_PER_FRAME = 1   # 每轮最多对几个行人做人脸（1=只处理最大框）
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_IMGSZ = 320                # 推理边长，越小越快（320/416/640）
YOLO_CONF = 0.45
TORCH_NUM_THREADS = 4           # RK3588 可试 4~6
DISPLAY_TARGET_FPS = 30       # 带窗口时的目标显示帧率
ASYNC_CAMERA_DEFAULT = True   # 摄像头+窗口默认异步（显示与推理分离）

# 串口门禁（香橙派识别到库内人员发 open，丢失后发 close）
SERIAL_DEVICE = "auto"          # auto | /dev/ttyACM0 | /dev/serial/by-id/...
SERIAL_BAUD = 115200
SERIAL_LINE_ENDING = "\n"       # 部分控制器要 "\r\n"
SERIAL_OPEN_CMD = "open"
SERIAL_CLOSE_CMD = "close"
SERIAL_OPEN_STABLE_SEC = 0.35   # 连续识别多久才发 open（越小越快，过小会抖动）
SERIAL_CLOSE_DELAY_SEC = 0.8    # 人离开后多久发 close
SERIAL_MIN_OPEN_HOLD_SEC = 0.5  # 开门后至少保持多久（给电机启动时间）
SERIAL_ENABLED_DEFAULT = True   # Linux 默认启用串口输出

SAMPLE_VIDEO_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
    "person-bicycle-car-detection.mp4"
)
SAMPLE_VIDEO_PATH = os.path.join(VIDEO_DIR, "person-detection.mp4")

for d in (DATA_DIR, FACE_DB_DIR, VIDEO_DIR, OUTPUT_DIR, ALERT_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)
