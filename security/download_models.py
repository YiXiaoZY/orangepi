"""
预下载安防所需模型（香橙派上建议先跑通本脚本再 run_security）

  python3 -m security.download_models
"""
from security.model_assets import ensure_all_models


def main():
    print("开始下载/检查模型（网络不稳可多试几次）...\n")
    ensure_all_models(verbose=True, facenet=True)
    print("\n全部完成。可执行:")
    print("  python3 -m security.run_security --camera --device 0")


if __name__ == "__main__":
    main()
