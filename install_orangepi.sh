#!/bin/bash
# 香橙派 5 安防项目 — 系统依赖一键安装（在板子上执行）
# 用法（必须在项目目录执行，不要逐段复制到终端）:
#   cd ~/test/te && chmod +x install_orangepi.sh && ./install_orangepi.sh

set -e

# 脚本所在目录；交互式粘贴时 $0 是 bash，会误指向 /usr/bin
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    PROJECT_DIR="$(pwd)"
fi
if [ ! -f "$PROJECT_DIR/requirements-orangepi.txt" ]; then
    echo "错误: 未在项目根目录找到 requirements-orangepi.txt"
    echo "请先: cd ~/test/te   再执行 ./install_orangepi.sh"
    exit 1
fi

echo "=== 1. 系统包 ==="
sudo apt update
# 勿与 v4l-utils 同批安装：香橙派镜像常 hold 该包，会导致整批失败
sudo apt install -y \
    python3 python3-pip python3-venv \
    python3-opencv \
    libgl1 libglib2.0-0 \
    git

if command -v v4l2-ctl >/dev/null 2>&1; then
    echo "v4l-utils 已可用，跳过"
else
    echo "尝试安装 v4l-utils（可选，失败可忽略）..."
    sudo apt install -y v4l-utils 2>/dev/null || \
        echo "提示: v4l-utils 被 hold，已有 v4l2-ctl 则无需处理"
fi

echo "=== 2. 用户加入 video 组（USB 摄像头权限）==="
sudo usermod -aG video "$USER" 2>/dev/null || true
echo "若首次加入 video 组，请注销后重新登录"

echo "=== 3. Python 虚拟环境（项目目录: $PROJECT_DIR）==="
bash "$PROJECT_DIR/setup_venv.sh"
# shellcheck disable=SC1091
source "$PROJECT_DIR/venv/bin/activate"
VENV="$PROJECT_DIR/venv"

echo "=== 4. 升级 pip ==="
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== 5. PyTorch (CPU / ARM64) ==="
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
    -i https://pypi.tuna.tsinghua.edu.cn/simple || {
    echo "PyTorch 官方源失败，尝试清华镜像..."
    pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
}

echo "=== 6. 项目依赖 ==="
pip install -r "$PROJECT_DIR/requirements-orangepi.txt" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== 7. 验证 ==="
python3 -c "import cv2; print('OpenCV', cv2.__version__)"
python3 -c "import torch; print('PyTorch', torch.__version__)"
python3 -c "import ultralytics; print('Ultralytics OK')"
python3 -c "import mediapipe; print('MediaPipe', mediapipe.__version__)"

echo ""
echo "=== 完成 ==="
echo "测试 USB:  python3 test_usb_camera.py --list-only"
echo "预览相机:  python3 usb_camera.py"
echo "下载模型:  python3 -m security.download_models"
echo "注册人脸:  python3 -m security.enroll"
echo "启动安防:  python3 -m security.run_security --camera --device 0"
echo ""
echo "若使用 venv，以后先执行: source $VENV/bin/activate"
