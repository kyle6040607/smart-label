"""MySQL 版資料存取層（Repository）。

實作與 app.repository.Repository 完全相同的一組方法，上層 API 與
pipeline 不用改任何一行就能切換（見 app/__init__.py 的後端選擇）。

連線目標由 Config 提供，支援兩種模式：
- TCP：MYSQL_HOST / MYSQL_PORT（本機 docker、Cloud SQL Public IP、Auth Proxy）
- Unix socket：MYSQL_UNIX_SOCKET=/cloudsql/<PROJECT>:<REGION>:<INSTANCE>
  （Cloud Run / App Engine 掛 Cloud SQL 的標準走法）

每個執行緒各持一條連線（threading.local），操作前 ping(reconnect=True)
自動處理 Cloud SQL 閒置斷線。多筆寫入包在同一個交易裡；LINE session
的狀態機用 SELECT ... FOR UPDATE 保證多 worker 下的原子性——這點比
JSON 版的 process 內鎖更強。
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor

from app.config import Config
from app.models import ImageRecord, Segment, LabelExample, User, LineSession

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id VARCHAR(32) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL DEFAULT '',
    path VARCHAR(512) NOT NULL DEFAULT '',
    width INT NOT NULL DEFAULT 0,
    height INT NOT NULL DEFAULT 0,
    file_hash VARCHAR(64) NOT NULL DEFAULT '',
    created_at DOUBLE NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS segments (
    id VARCHAR(32) PRIMARY KEY,
    image_id VARCHAR(32) NOT NULL,
    mask_path VARCHAR(512) NOT NULL DEFAULT '',
    bbox JSON NOT NULL,
    area INT NOT NULL DEFAULT 0,
    predicted_label VARCHAR(64) NULL,
    probs JSON NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 0,
    needs_review TINYINT(1) NOT NULL DEFAULT 0,
    human_label VARCHAR(64) NULL,
    reviewed TINYINT(1) NOT NULL DEFAULT 0,
    INDEX idx_segments_image_id (image_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS examples (
    id VARCHAR(32) PRIMARY KEY,
    label VARCHAR(64) NOT NULL,
    feature JSON NOT NULL,
    source_segment_id VARCHAR(32) NULL,
    created_at DOUBLE NOT NULL,
    INDEX idx_examples_label (label)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(32) PRIMARY KEY,
    username VARCHAR(64) NOT NULL DEFAULT '',
    password_hash VARCHAR(255) NOT NULL DEFAULT '',
    role VARCHAR(16) NOT NULL DEFAULT 'user',
    created_at DOUBLE NOT NULL,
    line_user_id VARCHAR(64) NULL,
    display_name VARCHAR(255) NOT NULL DEFAULT '',
    avatar_url VARCHAR(512) NOT NULL DEFAULT '',
    INDEX idx_users_username (username),
    INDEX idx_users_line_user_id (line_user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS line_sessions (
    line_user_id VARCHAR(64) PRIMARY KEY,
    image_ids JSON NOT NULL,
    images_done TINYINT(1) NOT NULL DEFAULT 0,
    confirmed TINYINT(1) NOT NULL DEFAULT 0,
    prompt TEXT NULL,
    updated_at DOUBLE NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


# ---------- row <-> dataclass ----------

def _row_to_image(r: dict) -> ImageRecord:
    return ImageRecord(
        id=r["id"], filename=r["filename"], path=r["path"],
        width=r["width"], height=r["height"], file_hash=r["file_hash"],
        created_at=r["created_at"],
    )


def _row_to_segment(r: dict) -> Segment:
    return Segment(
        id=r["id"], image_id=r["image_id"], mask_path=r["mask_path"],
        bbox=tuple(json.loads(r["bbox"])), area=r["area"],
        predicted_label=r["predicted_label"],
        probs=json.loads(r["probs"]), confidence=r["confidence"],
        needs_review=bool(r["needs_review"]),
        human_label=r["human_label"], reviewed=bool(r["reviewed"]),
    )


def _row_to_example(r: dict) -> LabelExample:
    return LabelExample(
        id=r["id"], label=r["label"], feature=json.loads(r["feature"]),
        source_segment_id=r["source_segment_id"], created_at=r["created_at"],
    )


def _row_to_user(r: dict) -> User:
    return User(
        id=r["id"], username=r["username"], password_hash=r["password_hash"],
        role=r["role"], created_at=r["created_at"],
        line_user_id=r["line_user_id"], display_name=r["display_name"],
        avatar_url=r["avatar_url"],
    )


def _row_to_session(r: dict) -> LineSession:
    return LineSession(
        line_user_id=r["line_user_id"], image_ids=json.loads(r["image_ids"]),
        images_done=bool(r["images_done"]), confirmed=bool(r["confirmed"]),
        prompt=r["prompt"], updated_at=r["updated_at"],
    )


class MySQLRepository:
    SESSION_TTL_SECONDS = 600

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._local = threading.local()
        self._ensure_schema()

    # ---------- 連線 / 交易 ----------
    def _connect(self) -> pymysql.connections.Connection:
        cfg = self._cfg
        kwargs: dict = dict(
            user=cfg.mysql_user,
            password=cfg.mysql_password,
            database=cfg.mysql_database,
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=False,
        )
        if cfg.mysql_unix_socket:
            kwargs["unix_socket"] = cfg.mysql_unix_socket
        else:
            kwargs["host"] = cfg.mysql_host
            kwargs["port"] = cfg.mysql_port
        return pymysql.connect(**kwargs)

    def _conn(self) -> pymysql.connections.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        else:
            conn.ping(reconnect=True)
        return conn

    @contextmanager
    def _tx(self):
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _ensure_schema(self) -> None:
        with self._tx() as cur:
            for stmt in _SCHEMA.split(";"):
                if stmt.strip():
                    cur.execute(stmt)

    # ---------- 影像 ----------
    def add_image(self, img: ImageRecord) -> ImageRecord:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO images (id, filename, path, width, height, file_hash, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (img.id, img.filename, img.path, img.width, img.height,
                 img.file_hash, img.created_at),
            )
        return img

    def get_image(self, image_id: str) -> ImageRecord | None:
        with self._tx() as cur:
            cur.execute("SELECT * FROM images WHERE id=%s", (image_id,))
            r = cur.fetchone()
        return _row_to_image(r) if r else None

    def list_images(self) -> list[ImageRecord]:
        with self._tx() as cur:
            cur.execute("SELECT * FROM images ORDER BY created_at")
            rows = cur.fetchall()
        return [_row_to_image(r) for r in rows]

    def delete_image(self, image_id: str) -> list[str]:
        """刪除一張圖，連帶清掉它所有的遮罩片段紀錄；回傳要刪的檔案路徑。"""
        with self._tx() as cur:
            cur.execute("SELECT path FROM images WHERE id=%s FOR UPDATE", (image_id,))
            img = cur.fetchone()
            if img is None:
                return []
            paths = [img["path"]]
            cur.execute("SELECT mask_path FROM segments WHERE image_id=%s", (image_id,))
            paths += [r["mask_path"] for r in cur.fetchall()]
            cur.execute("DELETE FROM segments WHERE image_id=%s", (image_id,))
            cur.execute("DELETE FROM images WHERE id=%s", (image_id,))
        return [p for p in paths if p]

    # ---------- 遮罩片段 ----------
    def _write_segment(self, cur, seg: Segment) -> None:
        cur.execute(
            "REPLACE INTO segments (id, image_id, mask_path, bbox, area,"
            " predicted_label, probs, confidence, needs_review, human_label, reviewed)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (seg.id, seg.image_id, seg.mask_path, json.dumps(list(seg.bbox)),
             seg.area, seg.predicted_label, json.dumps(seg.probs),
             seg.confidence, seg.needs_review, seg.human_label, seg.reviewed),
        )

    def add_segment(self, seg: Segment) -> Segment:
        with self._tx() as cur:
            self._write_segment(cur, seg)
        return seg

    def get_segment(self, seg_id: str) -> Segment | None:
        with self._tx() as cur:
            cur.execute("SELECT * FROM segments WHERE id=%s", (seg_id,))
            r = cur.fetchone()
        return _row_to_segment(r) if r else None

    def list_segments(self, image_id: str | None = None) -> list[Segment]:
        with self._tx() as cur:
            if image_id is None:
                cur.execute("SELECT * FROM segments")
            else:
                cur.execute("SELECT * FROM segments WHERE image_id=%s", (image_id,))
            rows = cur.fetchall()
        return [_row_to_segment(r) for r in rows]

    def list_review_queue(self) -> list[Segment]:
        with self._tx() as cur:
            cur.execute("SELECT * FROM segments WHERE needs_review=1 AND reviewed=0")
            rows = cur.fetchall()
        return [_row_to_segment(r) for r in rows]

    def delete_segment(self, seg_id: str) -> str | None:
        with self._tx() as cur:
            cur.execute("SELECT mask_path FROM segments WHERE id=%s FOR UPDATE", (seg_id,))
            r = cur.fetchone()
            if r is None:
                return None
            cur.execute("DELETE FROM segments WHERE id=%s", (seg_id,))
        return r["mask_path"] or None

    def update_segment(self, seg: Segment) -> Segment:
        with self._tx() as cur:
            self._write_segment(cur, seg)
        return seg

    # ---------- few-shot 範例 ----------
    def add_example(self, ex: LabelExample) -> LabelExample:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO examples (id, label, feature, source_segment_id, created_at)"
                " VALUES (%s, %s, %s, %s, %s)",
                (ex.id, ex.label, json.dumps(ex.feature),
                 ex.source_segment_id, ex.created_at),
            )
        return ex

    def list_examples(self) -> list[LabelExample]:
        with self._tx() as cur:
            cur.execute("SELECT * FROM examples")
            rows = cur.fetchall()
        return [_row_to_example(r) for r in rows]

    def labels(self) -> list[str]:
        with self._tx() as cur:
            cur.execute("SELECT DISTINCT label FROM examples ORDER BY label")
            rows = cur.fetchall()
        return [r["label"] for r in rows]

    def delete_label(self, label: str) -> int:
        """刪掉某類別的所有種子範例，連帶把該類別的人工標記退回送審。"""
        with self._tx() as cur:
            deleted = cur.execute("DELETE FROM examples WHERE label=%s", (label,))
            cur.execute(
                "UPDATE segments SET human_label=NULL, reviewed=0, needs_review=1"
                " WHERE human_label=%s",
                (label,),
            )
        return deleted

    # ---------- 使用者 / 登入 ----------
    def _write_user(self, cur, user: User) -> None:
        cur.execute(
            "REPLACE INTO users (id, username, password_hash, role, created_at,"
            " line_user_id, display_name, avatar_url)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (user.id, user.username, user.password_hash, user.role,
             user.created_at, user.line_user_id, user.display_name, user.avatar_url),
        )

    def add_user(self, user: User) -> User:
        with self._tx() as cur:
            self._write_user(cur, user)
        return user

    def get_user(self, user_id: str) -> User | None:
        with self._tx() as cur:
            cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
            r = cur.fetchone()
        return _row_to_user(r) if r else None

    def get_user_by_username(self, username: str) -> User | None:
        with self._tx() as cur:
            cur.execute("SELECT * FROM users WHERE username=%s LIMIT 1", (username,))
            r = cur.fetchone()
        return _row_to_user(r) if r else None

    def get_user_by_line_id(self, line_user_id: str) -> User | None:
        if not line_user_id:
            return None
        with self._tx() as cur:
            cur.execute("SELECT * FROM users WHERE line_user_id=%s LIMIT 1", (line_user_id,))
            r = cur.fetchone()
        return _row_to_user(r) if r else None

    def update_user(self, user: User) -> User:
        with self._tx() as cur:
            self._write_user(cur, user)
        return user

    def list_users(self) -> list[User]:
        with self._tx() as cur:
            cur.execute("SELECT * FROM users")
            rows = cur.fetchall()
        return [_row_to_user(r) for r in rows]

    # ---------- LINE session ----------
    def _write_session(self, cur, s: LineSession) -> None:
        cur.execute(
            "REPLACE INTO line_sessions (line_user_id, image_ids, images_done,"
            " confirmed, prompt, updated_at) VALUES (%s, %s, %s, %s, %s, %s)",
            (s.line_user_id, json.dumps(s.image_ids), s.images_done,
             s.confirmed, s.prompt, s.updated_at),
        )

    def _select_session(self, cur, line_user_id: str, for_update: bool = False) -> LineSession | None:
        sql = "SELECT * FROM line_sessions WHERE line_user_id=%s"
        if for_update:
            sql += " FOR UPDATE"
        cur.execute(sql, (line_user_id,))
        r = cur.fetchone()
        return _row_to_session(r) if r else None

    def get_line_session(self, line_user_id: str) -> LineSession | None:
        with self._tx() as cur:
            return self._select_session(cur, line_user_id)

    def add_line_session_image(self, line_user_id: str, image_id: str) -> tuple[LineSession, list[str], bool]:
        """新增一張圖到 session；回傳 (session, 逾時被清的舊圖 id, reopened)。"""
        with self._tx() as cur:
            s = self._select_session(cur, line_user_id, for_update=True)
            expired_ids: list[str] = []
            if s is not None and time.time() - s.updated_at > self.SESSION_TTL_SECONDS:
                expired_ids = list(s.image_ids)
                s = None
            if s is None:
                s = LineSession(line_user_id=line_user_id)
            reopened = s.images_done
            s.image_ids.append(image_id)
            s.images_done = False
            s.confirmed = False
            s.updated_at = time.time()
            self._write_session(cur, s)
            return s, expired_ids, reopened

    def mark_line_session_images_done(self, line_user_id: str) -> LineSession | None:
        with self._tx() as cur:
            s = self._select_session(cur, line_user_id, for_update=True)
            if s is None or not s.image_ids:
                return None
            s.images_done = True
            s.confirmed = False
            s.updated_at = time.time()
            self._write_session(cur, s)
            return s

    def confirm_line_session_images(self, line_user_id: str) -> LineSession | None:
        with self._tx() as cur:
            s = self._select_session(cur, line_user_id, for_update=True)
            if s is None or not s.images_done or s.confirmed:
                return None
            s.confirmed = True
            s.updated_at = time.time()
            self._write_session(cur, s)
            return s

    def reset_line_session_images(self, line_user_id: str) -> list[str]:
        with self._tx() as cur:
            s = self._select_session(cur, line_user_id, for_update=True)
            if s is None:
                return []
            old_ids = list(s.image_ids)
            s.image_ids = []
            s.images_done = False
            s.confirmed = False
            s.prompt = None
            s.updated_at = time.time()
            self._write_session(cur, s)
            return old_ids

    def set_line_session_prompt(self, line_user_id: str, prompt: str) -> LineSession | None:
        with self._tx() as cur:
            s = self._select_session(cur, line_user_id, for_update=True)
            if s is None or not s.confirmed:
                return None
            s.prompt = prompt
            s.updated_at = time.time()
            self._write_session(cur, s)
            return s

    def try_consume_line_session(self, line_user_id: str) -> tuple[str, LineSession | None]:
        """圖文都到齊時原子性地取出並刪除 session，避免多 worker 重複觸發。"""
        with self._tx() as cur:
            s = self._select_session(cur, line_user_id, for_update=True)
            if s is None or not s.image_ids:
                return "empty", None
            if not s.images_done:
                return "collecting_images", s
            if not s.confirmed:
                return "awaiting_confirmation", s
            if s.prompt is None:
                return "waiting_prompt", s
            cur.execute("DELETE FROM line_sessions WHERE line_user_id=%s", (line_user_id,))
            return "ready", s

    def clear_line_session(self, line_user_id: str) -> None:
        with self._tx() as cur:
            cur.execute("DELETE FROM line_sessions WHERE line_user_id=%s", (line_user_id,))

    # ---------- 統計 ----------
    def stats(self) -> dict:
        with self._tx() as cur:
            cur.execute(
                "SELECT COUNT(*) AS total,"
                " COALESCE(SUM(needs_review), 0) AS need_review,"
                " COALESCE(SUM(reviewed), 0) AS reviewed"
                " FROM segments"
            )
            seg = cur.fetchone()
            cur.execute("SELECT COUNT(*) AS n, COUNT(DISTINCT label) AS k FROM examples")
            ex = cur.fetchone()
        total = int(seg["total"])
        need_review = int(seg["need_review"])
        auto_accepted = total - need_review
        return {
            "total_segments": total,
            "auto_accepted": auto_accepted,
            "need_review": need_review,
            "reviewed": int(seg["reviewed"]),
            "auto_ratio": round(auto_accepted / total, 3) if total else 0.0,
            "num_examples": int(ex["n"]),
            "num_labels": int(ex["k"]),
        }
