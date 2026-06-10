import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from security.config import DB_PATH, FACE_DB_DIR, MATCH_THRESHOLD


def cosine_sim(a, b):
    return float(np.dot(a, b))


class FaceDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()
        self._cache = self._load_cache()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    track_id INTEGER,
                    alert_type TEXT NOT NULL,
                    person_name TEXT,
                    similarity REAL,
                    image_path TEXT
                )
                """
            )

    def _load_cache(self):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT p.name, e.vector
                FROM embeddings e
                JOIN persons p ON p.id = e.person_id
                """
            ).fetchall()
        cache = {}
        for name, blob in rows:
            vec = pickle.loads(blob)
            cache.setdefault(name, []).append(vec)
        return cache

    def add_person_embedding(self, name, vector):
        blob = pickle.dumps(vector.astype(np.float32))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR IGNORE INTO persons(name) VALUES (?)", (name,))
            person_id = conn.execute(
                "SELECT id FROM persons WHERE name=?", (name,)
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO embeddings(person_id, vector) VALUES (?, ?)",
                (person_id, blob),
            )
        self._cache.setdefault(name, []).append(vector)

    def match(self, vector, threshold=MATCH_THRESHOLD):
        """返回 (name, score) 或 (None, best_score)"""
        best_name, best_score = None, -1.0
        for name, vectors in self._cache.items():
            for ref in vectors:
                score = cosine_sim(vector, ref)
                if score > best_score:
                    best_score = score
                    best_name = name
        if best_score >= threshold:
            return best_name, best_score
        return None, best_score

    def log_alert(self, track_id, alert_type, person_name, similarity, image_path):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO alerts(ts, track_id, alert_type, person_name, similarity, image_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    track_id,
                    alert_type,
                    person_name,
                    similarity,
                    image_path,
                ),
            )

    def list_persons(self):
        return list(self._cache.keys())

    def count(self):
        return sum(len(v) for v in self._cache.values())


def enroll_from_folders(embedder, db=None):
    """扫描 data/face_db/姓名/*.jpg 注册人脸"""
    db = db or FaceDatabase()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    enrolled = 0

    for person_dir in sorted(Path(FACE_DB_DIR).iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        for img_path in person_dir.iterdir():
            if img_path.suffix.lower() not in exts:
                continue
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  跳过无效图片: {img_path}")
                continue
            vec = embedder.embed(img)
            if vec is None:
                print(f"  未检测到人脸: {img_path}")
                continue
            db.add_person_embedding(name, vec)
            enrolled += 1
            print(f"  已注册: {name} <- {img_path.name}")

    print(f"注册完成，共 {enrolled} 条特征，{len(db.list_persons())} 人")
    return db
