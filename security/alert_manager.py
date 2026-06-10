import time
from datetime import datetime
from pathlib import Path

import cv2

from security.config import ALERT_DIR, ALERT_COOLDOWN_SEC


class AlertManager:
    def __init__(self, cooldown=ALERT_COOLDOWN_SEC):
        self.cooldown = cooldown
        self._last_alert = {}  # track_id -> timestamp

    def should_alert(self, track_id):
        now = time.time()
        last = self._last_alert.get(track_id, 0)
        if now - last < self.cooldown:
            return False
        self._last_alert[track_id] = now
        return True

    def save_snapshot(self, frame, track_id, label):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{label}_id{track_id}_{ts}.jpg"
        path = Path(ALERT_DIR) / filename
        cv2.imwrite(str(path), frame)
        return str(path)
