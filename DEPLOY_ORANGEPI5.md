# 香橙派 5 部署清单（安防项目）

适用于：Orange Pi 5 + USB 摄像头 + 本项目 `security/` 模块。

---

## 一、硬件与系统

| 项目 | 建议 |
|------|------|
| 板子 | Orange Pi 5 (RK3588) |
| 系统 | Orange Pi OS / Debian / Ubuntu（64 位） |
| 摄像头 | USB UVC 摄像头（已验证 `/dev/video0`） |
| 存储 | 建议 32GB+，模型与依赖约 2~5GB |
| 网络 | 首次需联网下载模型 |

---

## 二、拷贝项目到板子

在 PC 上打包（或在板子 `git clone`）：

```bash
# 至少需要这些目录/文件
te/
├── security/              # 安防核心
├── data/face_db/          # 人脸库照片（按姓名分子目录）
├── models/                # 可空，首次运行自动下载
├── test_usb_camera.py     # USB 测试
├── usb_camera.py          # 相机预览/拍照
├── install_orangepi.sh
├── requirements-orangepi.txt
└── DEPLOY_ORANGEPI5.md
```

拷到板子示例路径：`/home/orangepi/te`

---

## 三、系统依赖（apt）

```bash
sudo apt update
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    git

# v4l-utils 单独装（香橙派镜像常 apt hold，勿与上面同批）
command -v v4l2-ctl >/dev/null || sudo apt install -y v4l-utils
```

| 包 | 作用 |
|----|------|
| `python3-opencv` | OpenCV（`cv2`），与系统 Python 绑定 |
| `v4l-utils` | `v4l2-ctl` 查看摄像头 |
| `libgl1` | OpenCV 显示窗口（无头可省略） |

---

## 四、USB 摄像头权限

```bash
ls -l /dev/video*          # 确认有 video0
sudo usermod -aG video orangepi
# 注销重新登录，或 reboot
groups | grep video
```

---

## 五、一键安装 Python 依赖

**不要**把 `install_orangepi.sh` 里的 `echo "=== 3..."` 等片段复制到终端。交互式 shell 里 `$0` 是 `bash`，`PROJECT_DIR` 会变成 `/usr/bin`，出现 `Permission denied: '/usr/bin/venv'`。

**只执行整文件**，或下面「仅 venv」三条命令：

```bash
cd ~/test/te
bash setup_venv.sh
```

完整依赖安装：

```bash
cd ~/test/te
chmod +x install_orangepi.sh
./install_orangepi.sh
```

或手动安装（先 `cd` 到项目根目录）：

```bash
cd ~/test/te
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# PyTorch ARM64 CPU
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-orangepi.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 验证依赖

```bash
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "import torch; print(torch.__version__)"
python3 -c "from ultralytics import YOLO; print('YOLO OK')"
python3 -c "import mediapipe; print(mediapipe.__version__)"
```

---

## 六、USB 摄像头测试（已通过可跳过）

```bash
python3 test_usb_camera.py --list-only
python3 test_usb_camera.py --device 0
# 查看 outputs/usb_camera_test_0.jpg

python3 usb_camera.py --device 0 --width 640 --height 480
# 按 q 退出
```

---

## 七、注册人脸库

目录结构：

```
data/face_db/
├── 张三/
│   ├── 1.jpg
│   └── 2.jpg
└── 李四/
    └── 1.jpg
```

每人 2~5 张正脸，光线均匀。

```bash
cd /home/orangepi/te
python3 -m security.enroll
```

成功后会生成 `data/security.db`。

---

## 八、预下载模型（首次必做）

`run_security` 首次会拉取 YOLO / MediaPipe / FaceNet，香橙派访问 GitHub/Google 易报：

`Remote end closed connection without response`

**先单独下载（可多试几次）：**

```bash
cd ~/test/te
source venv/bin/activate
python3 -m security.download_models
```

确认存在：

```bash
ls -lh models/yolov8n.pt models/blaze_face_short_range.tflite
ls ~/.cache/torch/checkpoints/
```

**PC 下载后 scp 到板子（离线推荐）：**

| 文件 | 目标 |
|------|------|
| `yolov8n.pt` | `项目/models/yolov8n.pt` |
| `blaze_face_short_range.tflite` | `项目/models/` |
| **`20180402-114759-vggface2.pt`**（约 107MB，**必拷**） | `项目/models/20180402-114759-vggface2.pt` |

FaceNet 下载地址（PC 浏览器）:
https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt

```bash
scp 20180402-114759-vggface2.pt orangepi@orangepi5:~/Downloads/te/models/
```

或拷贝 `~/.cache/torch/checkpoints/` 整个目录到板子用户主目录下同名路径。

## 九、启动安防（USB 摄像头）

```bash
cd ~/test/te
python3 -m security.run_security --camera --device 0
```

| 参数 | 说明 |
|------|------|
| `--camera` | 使用 USB 摄像头 |
| `--device 0` | 对应 `/dev/video0` |
| `--no-show` | 无显示器时不弹窗（需后续支持保存/日志） |

首次运行会自动下载：

- `yolov8n.pt`
- `models/blaze_face_short_range.tflite`

---

## 十、性能优化（香橙派）

瓶颈大致为：**YOLO 每帧** > **FaceNet+MediaPipe** > **imshow 显示**。

### 带窗口 30 FPS（默认异步模式）

显示与推理分离：**窗口尽量 30 帧**，YOLO+人脸在后台跑（识别流程不变）。

```bash
python3 -m security.run_security --camera --device 0 --profile \
  --width 640 --height 480 --imgsz 320 --face-interval 8 --display-fps 30
```

画面左上角：`显示:30 推理:12` — 显示接近 30，推理 10~15 属正常。

若仍卡顿，把 `--imgsz` 改为 `256`，或 `--face-interval 12`。

旧版「推理完才显示」：`--sync`（会明显变慢）。

### 无窗口最快

```bash
python3 -m security.run_security --camera --device 0 --no-show \
  --imgsz 320 --face-interval 15
```

| 参数 | 作用 | 建议 |
|------|------|------|
| `--width/height` | 摄像头采集分辨率 | 640×480，勿 1080p |
| `--imgsz` | YOLO 推理尺寸 | 320（更快）→ 416 → 640 |
| `--face-interval` | 每 N 帧做人脸 | 10~20，越大越流畅 |
| `--max-face` | 每轮最多识别几人 | 1（只认最大框行人） |
| `--no-show` | 不弹 OpenCV 窗口 | SSH/无桌面必加，可明显提速 |
| `--profile` | 显示 FPS | 调参时用 |

也可改 `security/config.py` 默认值，无需每次带参数。

### 预期帧率（参考）

| 配置 | 大致 FPS |
|------|----------|
| 默认 + 窗口显示 | 2~5 |
| `--imgsz 320 --face-interval 15 --no-show` | 5~12 |
| 仅 YOLO 跟踪、人脸间隔 30 | 8~15 |

人脸识别有缓存：已识别的 track 在非间隔帧不再跑 FaceNet。

---

## 十一、开机自启（可选）

创建 systemd 服务 `/etc/systemd/system/te-security.service`：

```ini
[Unit]
Description=TE Security Camera
After=network.target

[Service]
Type=simple
User=orangepi
WorkingDirectory=/home/orangepi/te
Environment=OPENCV_VIDEOIO_PRIORITY_LIST=V4L2
ExecStart=/home/orangepi/te/venv/bin/python3 -m security.run_security --camera --device 0 --no-show
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable te-security
sudo systemctl start te-security
sudo systemctl status te-security
journalctl -u te-security -f
```

---

## 十二、常见问题

| 现象 | 处理 |
|------|------|
| `Remote end closed connection without response` | 先 `python3 -m security.download_models` 或从 PC scp 模型到 `models/` |
| `Permission denied: '/usr/bin/venv'` | 勿逐段粘贴脚本；`cd ~/test/te` 后执行 `./install_orangepi.sh` 或手动 `python3 -m venv venv` |
| `Held packages were changed` | 去掉 `v4l-utils` 后重装；或 `sudo apt install -y --allow-change-held-packages v4l-utils` |
| `No module named 'cv2'` | `sudo apt install python3-opencv` |
| 打不开 `/dev/video0` | 检查 `video` 组、`lsusb`、换 USB 口 |
| `/dev/video1` 警告 | 正常，只用 `video0` |
| PyTorch 安装失败 | 用 `install_orangepi.sh` 或换清华源 |
| MediaPipe 安装失败 | `pip install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 运行很慢 | 降分辨率、增大 `FACE_DETECT_INTERVAL` |
| 内存不足 | 关闭桌面，只用 `yolov8n` |

---

## 十三、部署检查表（打勾）

- [ ] 项目已拷贝到 `/home/orangepi/te`
- [ ] `apt` 系统依赖已安装
- [ ] 用户已在 `video` 组
- [ ] `test_usb_camera.py` 通过
- [ ] `pip` 依赖验证通过
- [ ] `data/face_db/` 已放人脸照片
- [ ] `security.enroll` 成功
- [ ] `security.download_models` 成功（或已从 PC 拷贝模型）
- [ ] `run_security --camera` 能跑
- [ ] （可选）systemd 自启配置

---

## 十四、与 PC 开发环境差异

| PC (Windows) | 香橙派 5 |
|--------------|----------|
| CUDA 可选 | 一般 CPU 推理 |
| OpenCV 4.13 | apt 可能 4.5.x，够用 |
| 高 FPS | 3~10 FPS，需降负载 |

开发在 PC，部署在香橙派时，只需同步代码 + `data/face_db/`，不必拷贝 PC 的 `venv`。
