"""兼容入口 — 请优先使用 scripts/ssh_pi.py deploy"""
import subprocess
import sys

if __name__ == "__main__":
    script = __file__.replace("deploy_remote_to_pi.py", "ssh_pi.py")
    sys.exit(subprocess.call([sys.executable, script, "deploy"]))
