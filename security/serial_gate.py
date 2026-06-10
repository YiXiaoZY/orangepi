import os
import termios
import threading
import time
from glob import glob


class SerialGate:
    """识别到库内人员发 open，丢失并稳定一段时间后发 close"""

    def __init__(
        self,
        device="auto",
        baud=115200,
        line_ending="\n",
        open_cmd="open",
        close_cmd="close",
        open_stable_sec=0.8,
        close_delay_sec=2.0,
        min_open_hold_sec=1.5,
    ):
        self.device = device
        self.baud = baud
        self.line_ending = line_ending
        self.open_cmd = open_cmd
        self.close_cmd = close_cmd
        self.open_stable_sec = open_stable_sec
        self.close_delay_sec = close_delay_sec
        self.min_open_hold_sec = min_open_hold_sec
        self._fd = None
        self._last_payload = None
        self._state = "closed"  # closed | open
        self._known_since = None
        self._unknown_since = None
        self._opened_at = None
        self._lock = threading.Lock()
        self._open()
        self._emit(self.close_cmd)

    @staticmethod
    def resolve_device(device):
        if device and device != "auto":
            return device

        by_id = sorted(glob("/dev/serial/by-id/*"))
        for path in by_id:
            target = os.path.realpath(path)
            base = os.path.basename(target)
            if base.startswith(("ttyACM", "ttyUSB", "ttyS")):
                return path

        for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*", "/dev/ttyS*"):
            matches = sorted(glob(pattern))
            if matches:
                return matches[0]

        return "/dev/ttyACM0"

    def _open(self):
        self.device = self.resolve_device(self.device)
        try:
            self._fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            self._configure()
            print(
                f"串口已打开: {self.device} @ {self.baud} "
                f"(指令: '{self.open_cmd}' / '{self.close_cmd}')"
            )
        except OSError as e:
            self._fd = None
            print(f"警告: 无法打开串口 {self.device}: {e}")

    def _configure(self):
        baud_const = getattr(termios, f"B{self.baud}", None)
        if baud_const is None:
            raise OSError(f"不支持的波特率: {self.baud}")

        attrs = termios.tcgetattr(self._fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] |= termios.CLOCAL | termios.CREAD
        attrs[2] &= ~termios.PARENB
        attrs[2] &= ~termios.CSTOPB
        attrs[2] &= ~termios.CSIZE
        attrs[2] |= termios.CS8
        if hasattr(termios, "CRTSCTS"):
            attrs[2] &= ~termios.CRTSCTS
        attrs[3] = 0
        attrs[4] = baud_const
        attrs[5] = baud_const
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self._fd, termios.TCSANOW, attrs)

    def _emit(self, payload: str) -> bool:
        with self._lock:
            if self._fd is None or payload == self._last_payload:
                return False
            data = f"{payload}{self.line_ending}".encode("ascii", errors="ignore")
            try:
                os.write(self._fd, data)
                self._last_payload = payload
                print(f"[串口] -> {payload!r} ({len(data)} bytes)")
                return True
            except OSError as e:
                print(f"警告: 串口发送失败 {payload!r}: {e}")
                self._close_fd()
                return False

    def send_raw(self, text: str) -> bool:
        """手动测试用，发送任意字符串"""
        return self._emit(text)

    def force_open(self):
        self._state = "open"
        self._opened_at = time.monotonic()
        self._known_since = self._opened_at
        self._unknown_since = None
        self._emit(self.open_cmd)

    def force_close(self):
        self._state = "closed"
        self._known_since = None
        self._unknown_since = None
        self._opened_at = None
        self._emit(self.close_cmd)

    def update_known_presence(self, known_present: bool):
        """带防抖：稳定识别后才 open，开门后至少保持 min_open_hold_sec"""
        now = time.monotonic()

        if known_present:
            self._unknown_since = None
            if self._known_since is None:
                self._known_since = now
            if (
                self._state != "open"
                and now - self._known_since >= self.open_stable_sec
            ):
                self._state = "open"
                self._opened_at = now
                self._emit(self.open_cmd)
            return

        self._known_since = None
        if self._state != "open":
            if self._state != "closed":
                self._state = "closed"
                self._emit(self.close_cmd)
            return

        if self._unknown_since is None:
            self._unknown_since = now

        held = (now - self._opened_at) if self._opened_at else 0.0
        absent = now - self._unknown_since
        if held >= self.min_open_hold_sec and absent >= self.close_delay_sec:
            self._state = "closed"
            self._opened_at = None
            self._unknown_since = None
            self._emit(self.close_cmd)

    def _close_fd(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def status_dict(self) -> dict:
        return {
            "ok": self._fd is not None,
            "state": self._state,
            "device": self.device,
            "last_cmd": self._last_payload,
        }

    def close(self):
        if self._fd is not None and self._state == "open":
            self.force_close()
        self._close_fd()
