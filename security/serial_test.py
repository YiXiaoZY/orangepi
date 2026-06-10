"""
串口手动测试 — 排查电机/继电器是否响应

  python3 -m security.serial_test open
  python3 -m security.serial_test close
  python3 -m security.serial_test send --text "1"
  python3 -m security.serial_test send --text "ON\\r\\n"
"""
import argparse
import time

from security.config import (
    SERIAL_BAUD,
    SERIAL_CLOSE_CMD,
    SERIAL_CLOSE_DELAY_SEC,
    SERIAL_DEVICE,
    SERIAL_LINE_ENDING,
    SERIAL_MIN_OPEN_HOLD_SEC,
    SERIAL_OPEN_CMD,
    SERIAL_OPEN_STABLE_SEC,
)
from security.serial_gate import SerialGate


def main():
    parser = argparse.ArgumentParser(description="串口门禁手动测试")
    parser.add_argument("action", choices=["open", "close", "send", "pulse"])
    parser.add_argument("--device", default=SERIAL_DEVICE)
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    parser.add_argument("--text", help="send 动作的自定义文本（支持 \\r \\n）")
    parser.add_argument("--hold", type=float, default=3.0, help="pulse 保持秒数")
    args = parser.parse_args()

    gate = SerialGate(
        device=args.device,
        baud=args.baud,
        line_ending=SERIAL_LINE_ENDING,
        open_cmd=SERIAL_OPEN_CMD,
        close_cmd=SERIAL_CLOSE_CMD,
        open_stable_sec=SERIAL_OPEN_STABLE_SEC,
        close_delay_sec=SERIAL_CLOSE_DELAY_SEC,
        min_open_hold_sec=SERIAL_MIN_OPEN_HOLD_SEC,
    )

    try:
        if args.action == "open":
            gate.force_open()
        elif args.action == "close":
            gate.force_close()
        elif args.action == "send":
            if not args.text:
                parser.error("send 需要 --text")
            text = args.text.encode("utf-8").decode("unicode_escape")
            gate.send_raw(text)
        elif args.action == "pulse":
            gate.force_open()
            print(f"保持 {args.hold}s ...")
            time.sleep(args.hold)
            gate.force_close()
        print("完成。观察电机/继电器是否动作。")
    finally:
        gate.close()


if __name__ == "__main__":
    main()
