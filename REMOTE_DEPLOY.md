# 远程推理部署（香橙派显示 + PC 计算）

```
香橙派 (10.71.111.195)          你的 PC (局域网 IP)
  USB 摄像头                       YOLO + FaceNet + 人脸库
  采集画面 ──JPEG──HTTP POST──►   /api/infer 推理
  OpenCV 窗口 ◄──JSON 框/姓名──   返回 overlays
```

香橙派只做：**采图 → 上传 → 画框显示**（可稳定 30fps 窗口）  
PC 做：**检测、跟踪、人脸识别、告警**（可用更强 CPU/GPU）

---

## SSH 托管（推荐）

PC 一键管理香橙派 `~/Downloads/te`：

```powershell
cd D:\te
python scripts/ssh_pi.py setup-key   # 首次：配置免密
python scripts/ssh_pi.py deploy      # 同步代码到板子
python scripts/ssh_pi.py status      # 查看板子状态 + PC 服务连通
python scripts/ssh_pi.py client      # SSH 启动远程客户端
python scripts/ssh_pi.py run -- curl http://10.71.111.243:8765/health
```

环境变量（可选）：`PI_HOST` `PI_USER` `PI_PROJECT` `PC_SERVER`

---

## 一、PC 端（`d:\te`）

### 1. 安装依赖

```powershell
cd d:\te
conda activate face_detetor
pip install -r requirements-remote.txt
```

### 2. 人脸库在 PC 上维护

```powershell
# 照片放 data/face_db/姓名/*.jpg
python -m security.enroll
```

可把香橙派上的 `data/face_db/`、`data/security.db` 拷到 PC 同名目录。

### 3. 查 PC 局域网 IP

```powershell
ipconfig
# 记下 IPv4，例如 10.71.111.88
```

### 4. 启动推理服务

```powershell
cd d:\te
python -m security.remote_server --host 0.0.0.0 --port 8765
```

监控面板：`http://<你PC的IP>:8765/`（例如 http://10.71.111.243:8765/）

健康检查：`http://localhost:8765/health`

**Windows 防火墙**：首次弹窗允许 Python，或手动放行 **8765/TCP 入站**。

---

## 二、香橙派端（`~/Downloads/te`）

只需轻量依赖（不必装 torch/ultralytics）：

```bash
pip install requests opencv-python-headless
# 或已有 python3-opencv + pip install requests
```

同步到板子的文件：

- `security/remote_client.py`
- `security/pipeline.py`（仅用于 `open_camera`、画框）
- `security/config.py`

### 启动客户端

```bash
cd ~/Downloads/te
python3 -m security.remote_client \
  --server http://<你PC的IP>:8765 \
  --camera --device 0 --profile --display-fps 30
```

示例（PC IP 为 `10.71.111.88`）：

```bash
python3 -m security.remote_client --server http://10.71.111.88:8765 --camera --device 0
```

画面左上角：`显示:30 远程推理:15 45ms OK`

---

## 三、SSH 连接香橙派

```bash
ssh orangepi@10.71.111.195
# 密码: orangepi
```

NoMachine（端口 4000）用于远程桌面；SSH（端口 22）用于命令行。

---

## 四、调优

| 参数 | 位置 | 说明 |
|------|------|------|
| `--jpeg-quality 70` | 客户端 | 越小上传越快，画质略降 |
| `--display-fps 30` | 客户端 | 窗口刷新率 |
| `--imgsz` | PC `inference_engine` / config | 推理尺寸 |
| `FACE_DETECT_INTERVAL` | PC `config.py` | 人脸比对间隔 |

网络延迟约 20~80ms（同局域网），推理时间看 PC 性能。

---

## 五、常见问题

| 现象 | 处理 |
|------|------|
| 客户端连不上 | PC 服务是否启动；IP/端口是否正确；防火墙 |
| 显示有、无框 | 等几秒；看 PC 终端是否有 `[告警]` 输出 |
| 人脸库为空 | 在 **PC** 执行 `python -m security.enroll` |
| 识别慢 | 正常，远程推理 FPS 取决于 PC + 网络 |
