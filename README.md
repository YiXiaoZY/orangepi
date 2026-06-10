# Orange Pi 安防视觉系统

基于 **Orange Pi 5 + USB 摄像头** 的嵌入式安防方案：行人检测、多目标跟踪、人脸识别、陌生人告警，支持 **PC 远程推理**、**串口门禁** 与 **Web 监控面板**。

```
香橙派 (采集端)                    PC (推理端)
  USB 摄像头                         YOLOv8 + ByteTrack + FaceNet
  采图 ──JPEG──HTTP POST──►         /api/infer 推理
  OpenCV 窗口 ◄──JSON 框/姓名──     人脸库比对 + 告警
  串口 open/close ◄── 识别结果       监控面板 / 状态 API
```

---

## 功能特性

| 模块 | 说明 |
|------|------|
| 行人检测 | YOLOv8n 检测 + ByteTrack 跨帧跟踪 |
| 人脸识别 | FaceNet 特征提取，SQLite 人脸库 1:N 比对 |
| 陌生人告警 | 未匹配人员自动截图，冷却去重 |
| 远程推理 | 香橙派只负责采图/显示，PC 承担算力（局域网 ~30fps 显示） |
| 串口门禁 | 识别库内人员发 `open`，离开后发 `close`（防抖可配） |
| 监控面板 | PC 端纯状态 Web 页，无实时视频流 |
| 本地推理 | 也可在香橙派本机跑完整 pipeline（需安装 torch 等） |

---

## 技术栈

- **检测/跟踪**：Ultralytics YOLOv8、ByteTrack
- **人脸识别**：facenet-pytorch、MediaPipe BlazeFace
- **推理服务**：FastAPI + Uvicorn
- **嵌入式端**：OpenCV、pyserial（串口）
- **数据存储**：SQLite + 本地照片目录
- **学习示例**：`tensflaw/` 目录含 TensorFlow / MediaPipe 入门脚本

---

## 项目结构

```
te/
├── security/                 # 安防核心
│   ├── remote_server.py      # PC 推理服务 + 监控 API
│   ├── remote_client.py      # 香橙派薄客户端（采图/显示/心跳/串口）
│   ├── pipeline.py           # 本地完整推理流水线
│   ├── inference_engine.py   # 推理引擎（检测+跟踪+识别）
│   ├── face_database.py      # 人脸库管理
│   ├── serial_gate.py        # 串口门禁控制
│   ├── dashboard.html        # 监控面板页面
│   └── config.py             # 全局参数
├── scripts/
│   └── ssh_pi.py             # PC 一键部署/管理香橙派
├── models/                   # 模型权重（大文件走 Git LFS）
├── data/
│   ├── face_db/              # 人脸库照片（按姓名分子目录）
│   └── videos/               # 测试视频
├── tensflaw/                 # TensorFlow 学习示例
├── DEPLOY_ORANGEPI5.md       # 香橙派本地部署详细说明
└── REMOTE_DEPLOY.md          # 远程推理部署详细说明
```

---

## 快速开始

### 环境要求

| 端 | 硬件/系统 | Python |
|----|-----------|--------|
| PC | Windows / Linux，建议有 GPU 或较强 CPU | 3.10+，conda 环境 `face_detetor` |
| 香橙派 | Orange Pi 5，Orange Pi OS / Debian 64 位 | 3.10+ |

### 1. 克隆项目

```bash
git clone git@github.com:YiXiaoZY/orangepi.git
cd orangepi
```

### 2. 下载模型（首次必做）

```bash
# PC 或香橙派均可执行
python -m security.download_models
```

模型文件：`yolov8n.pt`、`blaze_face_short_range.tflite`、`20180402-114759-vggface2.pt`（约 107MB）。  
香橙派网络不稳时，可在 PC 下载后 `scp` 到板子 `models/` 目录。

### 3. 录入人脸库

将照片放入 `data/face_db/<姓名>/*.jpg`，然后：

```bash
python -m security.enroll
```

### 4. 方案 A：远程推理（推荐）

**PC 端：**

```powershell
conda activate face_detetor
pip install -r requirements-remote.txt
# 还需已安装：torch, ultralytics, facenet-pytorch, mediapipe, opencv 等

python -m security.remote_server --host 0.0.0.0 --port 8765
```

监控面板：`http://<PC局域网IP>:8765/`

**香橙派端：**

```bash
pip install requests opencv-python-headless

python3 -m security.remote_client \
  --server http://<PC_IP>:8765 \
  --camera --device 0 --profile --display-fps 30
```

**PC 一键同步代码到板子：**

```powershell
python scripts/ssh_pi.py setup-key   # 首次配置 SSH 免密
python scripts/ssh_pi.py deploy      # 同步代码
python scripts/ssh_pi.py status      # 查看连通状态
```

详见 [REMOTE_DEPLOY.md](REMOTE_DEPLOY.md)。

### 5. 方案 B：香橙派本地推理

```bash
pip install -r requirements-orangepi.txt
python3 -m security.run_security --camera --device 0
```

详见 [DEPLOY_ORANGEPI5.md](DEPLOY_ORANGEPI5.md)。

---

## 串口门禁

识别到库内人员且稳定一段时间后，向串口发送 `open`；人员离开后发 `close`。

```bash
# 手动测试串口
python3 -m security.serial_test pulse --hold 3
```

主要参数在 `security/config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SERIAL_DEVICE` | `auto` | 串口设备路径 |
| `SERIAL_OPEN_STABLE_SEC` | `0.35` | 连续识别多久才开门 |
| `SERIAL_CLOSE_DELAY_SEC` | `0.8` | 人离开后多久关门 |
| `SERIAL_MIN_OPEN_HOLD_SEC` | `0.5` | 开门最短保持时间 |

---

## API 接口（PC 推理服务）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 监控面板 |
| `/health` | GET | 健康检查 |
| `/api/infer` | POST | 上传 JPEG 帧，返回检测框与识别结果 |
| `/api/status` | GET | 服务状态 |
| `/api/heartbeat` | POST | 香橙派心跳上报 |

---

## 常见问题

| 现象 | 处理 |
|------|------|
| 模型下载超时 | 在 PC 执行 `download_models`，再 `scp` 到板子 |
| 客户端连不上 PC | 确认服务已启动、IP/端口正确、防火墙放行 8765 |
| 有画面无识别框 | 等人进入画面几秒；检查 PC 终端日志 |
| 人脸库为空 | 在 **PC** 执行 `python -m security.enroll` |
| 串口电机抖动 | 调大 `SERIAL_OPEN_STABLE_SEC` 和 `SERIAL_CLOSE_DELAY_SEC` |

---

## 许可证

本项目仅供学习与研究使用。模型权重遵循各自上游项目的许可协议。
