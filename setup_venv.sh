#!/bin/bash
# 仅创建并激活项目 venv（可单独执行，勿逐段粘贴 install_orangepi.sh）
# 用法: cd ~/test/te && bash setup_venv.sh

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -f "$ROOT/requirements-orangepi.txt" ]; then
    echo "错误: 当前目录不是项目根（缺少 requirements-orangepi.txt）"
    echo "请: cd ~/test/te"
    exit 1
fi

echo "项目目录: $ROOT"
python3 -m venv "$ROOT/venv"
echo ""
echo "venv 已创建: $ROOT/venv"
echo "请在本终端执行（激活环境）:"
echo "  source $ROOT/venv/bin/activate"
