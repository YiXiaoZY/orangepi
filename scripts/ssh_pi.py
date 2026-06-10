"""
香橙派 SSH 托管工具（PC 端执行）

  python scripts/ssh_pi.py setup-key     # 配置免密登录
  python scripts/ssh_pi.py deploy        # 同步代码到板子
  python scripts/ssh_pi.py status        # 查看板子项目状态
  python scripts/ssh_pi.py run -- <cmd>  # 远程执行命令
  python scripts/ssh_pi.py client        # 启动远程推理客户端（前台提示）
"""
from __future__ import annotations

import argparse
import os
import sys

import paramiko

HOST = os.environ.get("PI_HOST", "10.71.111.195")
USER = os.environ.get("PI_USER", "orangepi")
PASSWORD = os.environ.get("PI_PASSWORD", "orangepi")
REMOTE_DIR = os.environ.get("PI_PROJECT", "/home/orangepi/Downloads/te")
PC_SERVER = os.environ.get("PC_SERVER", "http://10.71.111.243:8765")

SYNC_FILES = [
    "security/__init__.py",
    "security/config.py",
    "security/pipeline.py",
    "security/remote_client.py",
    "security/run_security.py",
    "security/alert_manager.py",
    "security/face_database.py",
    "security/face_embedder.py",
    "security/model_assets.py",
    "security/enroll.py",
    "security/download_models.py",
    "security/serial_gate.py",
    "security/serial_test.py",
    # PC 端: dashboard.html, monitor_state.py, remote_server.py 无需同步到板子
    "test_usb_camera.py",
    "usb_camera.py",
    "setup_venv.sh",
    "requirements-orangepi.txt",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _key_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".ssh", "id_ed25519")


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key_file = _key_path()
    try:
        if os.path.isfile(key_file):
            client.connect(
                HOST, username=USER, key_filename=key_file, timeout=15,
            )
            return client
    except paramiko.SSHException:
        pass
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    return client


def run_remote(cmd: str, timeout: float = 60) -> tuple[int, str, str]:
    client = connect()
    try:
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        code = stdout.channel.recv_exit_status()
        return code, stdout.read().decode(), stderr.read().decode()
    finally:
        client.close()


def cmd_setup_key():
    pub = _key_path() + ".pub"
    if not os.path.isfile(pub):
        print("未找到公钥，请先: ssh-keygen -t ed25519")
        sys.exit(1)
    with open(pub, encoding="utf-8") as f:
        pubkey = f.read().strip()
    client = connect()
    try:
        cmds = (
            "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
            f"grep -qF '{pubkey.split()[1]}' ~/.ssh/authorized_keys 2>/dev/null || "
            f"echo '{pubkey}' >> ~/.ssh/authorized_keys && "
            "chmod 600 ~/.ssh/authorized_keys"
        )
        stdin, stdout, stderr = client.exec_command(cmds)
        code = stdout.channel.recv_exit_status()
        if code == 0:
            print(f"免密登录已配置: {USER}@{HOST}")
        else:
            print(stderr.read().decode(), file=sys.stderr)
            sys.exit(code)
    finally:
        client.close()
    code, out, _ = run_remote("echo key_ok && hostname")
    print(out.strip())


def cmd_deploy():
    client = connect()
    sftp = client.open_sftp()
    print(f"同步到 {USER}@{HOST}:{REMOTE_DIR}")
    for rel in SYNC_FILES:
        local = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.isfile(local):
            print(f"  跳过（本地不存在）: {rel}")
            continue
        remote = f"{REMOTE_DIR}/{rel}"
        remote_dir = os.path.dirname(remote)
        client.exec_command(f"mkdir -p {remote_dir}")[1].channel.recv_exit_status()
        sftp.put(local, remote)
        print(f"  OK {rel}")
    sftp.close()
    post = (
        f"cd {REMOTE_DIR} && "
        "pip3 install requests -q 2>/dev/null || python3 -m pip install requests -q; "
        "python3 -c 'import security.remote_client; print(\"import ok\")'"
    )
    code, out, err = run_remote(post)
    print(out, end="")
    if err:
        print(err, file=sys.stderr)
    client.close()
    print("部署完成。")


def cmd_status():
    cmd = (
        f"echo '=== host ===' && hostname && "
        f"echo '=== project ===' && ls -la {REMOTE_DIR} && "
        f"echo '=== security ===' && ls -la {REMOTE_DIR}/security/*.py && "
        f"echo '=== models ===' && ls -lh {REMOTE_DIR}/models/ && "
        f"echo '=== face_db ===' && ls -la {REMOTE_DIR}/data/face_db/ 2>/dev/null; "
        f"find {REMOTE_DIR}/data/face_db -name '*.jpg' 2>/dev/null | head -5 && "
        f"echo '=== pc health ===' && "
        f"curl -s --connect-timeout 3 {PC_SERVER}/health || echo 'PC服务未连通'"
    )
    code, out, err = run_remote(cmd)
    print(out)
    if err:
        print(err, file=sys.stderr)


def cmd_client():
    cmd = (
        f"cd {REMOTE_DIR} && "
        f"python3 -m security.remote_client "
        f"--server {PC_SERVER} --camera --device 0 --profile --display-fps 30 "
        f"--serial-device auto"
    )
    print(f"远程启动（需 NoMachine/桌面看窗口）:\n  {cmd}\n")
    print("正在连接 SSH 执行（Ctrl+C 中断）...")
    client = connect()
    try:
        channel = client.get_transport().open_session()
        channel.get_pty()
        channel.exec_command(cmd)
        while True:
            if channel.recv_ready():
                sys.stdout.write(channel.recv(4096).decode(errors="replace"))
                sys.stdout.flush()
            if channel.recv_stderr_ready():
                sys.stderr.write(channel.recv_stderr(4096).decode(errors="replace"))
                sys.stderr.flush()
            if channel.exit_status_ready():
                break
        sys.exit(channel.recv_exit_status())
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="香橙派 SSH 托管")
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("setup-key", help="配置 SSH 免密")
    sub.add_parser("deploy", help="同步项目代码到板子")
    sub.add_parser("status", help="查看板子状态")
    sub.add_parser("client", help="SSH 启动远程推理客户端")

    p_run = sub.add_parser("run", help="执行远程 shell 命令")
    p_run.add_argument("remote_cmd", nargs=argparse.REMAINDER, help="命令")

    args = parser.parse_args()
    if args.action == "setup-key":
        cmd_setup_key()
    elif args.action == "deploy":
        cmd_deploy()
    elif args.action == "status":
        cmd_status()
    elif args.action == "client":
        cmd_client()
    elif args.action == "run":
        cmd = " ".join(args.remote_cmd).strip()
        if not cmd:
            print("用法: python scripts/ssh_pi.py run -- ls -la")
            sys.exit(1)
        code, out, err = run_remote(cmd)
        print(out, end="")
        if err:
            print(err, file=sys.stderr)
        sys.exit(code)


if __name__ == "__main__":
    main()
