"""MySQL 版資料存取層（Repository）。

實作與 app.repository.Repository 完全相同的一組方法，上層 API 與
pipeline 不用改任何一行就能切換（見 app/__init__.py 的後端選擇）。

連線目標由 Config 提供，支援兩種模式：
- TCP：MYSQL_HOST / MYSQL_PORT（本機 docker、Cloud SQL Public IP、Auth Proxy）
- Unix socket：MYSQL_UNIX_SOCKET=/cloudsql/<PROJECT>:<REGION>:<INSTANCE>
  （Cloud Run / App Engine 掛 Cloud SQL 的標準走法）

每個執行緒各持一條連線（threading.local），操作前 ping(reconnect=True)
自動處理 Cloud SQL 閒置斷線，多筆寫入包在同一個交易裡。
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor

from app.config import Config
from app.models import AnnotationTask,ImageRecord, Segment, LabelExample, User

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id VARCHAR(32) PRIMARY KEY,
    owner_id VARCHAR(32) NOT NULL DEFAULT '',
    filename VARCHAR(255) NOT NULL DEFAULT '',
    path VARCHAR(512) NOT NULL DEFAULT '',
    width INT NOT NULL DEFAULT 0,
    height INT NOT NULL DEFAULT 0,
    file_hash VARCHAR(64) NOT NULL DEFAULT '',
    created_at DOUBLE NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS parameters (
    `key` VARCHAR(64) PRIMARY KEY,
    value FLOAT NOT NULL DEFAULT 0
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
    email VARCHAR(255) NOT NULL DEFAULT '',
    email_verified TINYINT(1) NOT NULL DEFAULT 0,
    otp_hash VARCHAR(255) NOT NULL DEFAULT '',
    otp_expires DOUBLE NOT NULL DEFAULT 0,
    otp_attempts INT NOT NULL DEFAULT 0,
    role VARCHAR(16) NOT NULL DEFAULT 'user',
    created_at DOUBLE NOT NULL,
    line_user_id VARCHAR(64) NULL,
    display_name VARCHAR(255) NOT NULL DEFAULT '',
    avatar_url VARCHAR(512) NOT NULL DEFAULT '',
    INDEX idx_users_username (username),
    INDEX idx_users_line_user_id (line_user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS annotation_tasks (
    id VARCHAR(32) PRIMARY KEY,
    user_id VARCHAR(32) NOT NULL DEFAULT '',
    line_user_id VARCHAR(64) NOT NULL DEFAULT '',
    prompt TEXT NOT NULL,
    image_ids JSON NOT NULL,
    processed_image_ids JSON NOT NULL,
    dataset_version INT NOT NULL DEFAULT 0,
    notified_dataset_version INT NOT NULL DEFAULT 0,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    dataset_zip_path VARCHAR(512) NOT NULL DEFAULT '',
    best_model_path VARCHAR(512) NOT NULL DEFAULT '',
    download_token VARCHAR(64) NOT NULL,
    error_message TEXT NOT NULL,
    created_at DOUBLE NOT NULL,
    updated_at DOUBLE NOT NULL,
    INDEX idx_annotation_tasks_user_id (user_id),
    INDEX idx_annotation_tasks_line_user_id (line_user_id),
    UNIQUE INDEX uq_annotation_tasks_download_token (
        download_token
    )
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;

"""


# ---------- row <-> dataclass ----------

def _row_to_image(r: dict) -> ImageRecord:
    return ImageRecord(
        id=r["id"], owner_id=r.get("owner_id", ""),
        filename=r["filename"], path=r["path"],
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
        email=r.get("email", ""),
        email_verified=bool(r.get("email_verified", 0)),
        otp_hash=r.get("otp_hash", ""),
        otp_expires=r.get("otp_expires", 0.0),
        otp_attempts=r.get("otp_attempts", 0),
        role=r["role"], created_at=r["created_at"],
        line_user_id=r["line_user_id"], display_name=r["display_name"],
        avatar_url=r["avatar_url"],
    )

def _row_to_task(
    row: dict,
) -> AnnotationTask:
    return AnnotationTask(
        id=row["id"],
        user_id=row["user_id"],
        line_user_id=row["line_user_id"],
        prompt=row["prompt"],
        image_ids=json.loads(row["image_ids"]),
        processed_image_ids=json.loads(row["processed_image_ids"]),
        dataset_version=int(row["dataset_version"]),
        notified_dataset_version=int(row["notified_dataset_version"]),
        status=row["status"],
        dataset_zip_path=row["dataset_zip_path"],
        best_model_path=row["best_model_path"],
        download_token=row["download_token"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )

class MySQLRepository:
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
            # 既有資料表的欄位遷移：舊表缺欄位時補上
            migrations = [
                ("email", "VARCHAR(255) NOT NULL DEFAULT '' AFTER password_hash"),
                ("email_verified", "TINYINT(1) NOT NULL DEFAULT 0 AFTER email"),
                ("otp_hash", "VARCHAR(255) NOT NULL DEFAULT '' AFTER email_verified"),
                ("otp_expires", "DOUBLE NOT NULL DEFAULT 0 AFTER otp_hash"),
                ("otp_attempts", "INT NOT NULL DEFAULT 0 AFTER otp_expires"),
            ]
            for column, ddl in migrations:
                cur.execute(f"SHOW COLUMNS FROM users LIKE '{column}'")
                if cur.fetchone() is None:
                    cur.execute(f"ALTER TABLE users ADD COLUMN {column} {ddl}")

            # 圖片擁有者：舊資料保留空值，僅管理者能在 Web 介面存取。
            cur.execute("SHOW COLUMNS FROM images LIKE 'owner_id'")
            if cur.fetchone() is None:
                cur.execute(
                    "ALTER TABLE images "
                    "ADD COLUMN owner_id VARCHAR(32) NOT NULL DEFAULT '' "
                    "AFTER id"
                )
                cur.execute(
                    "CREATE INDEX idx_images_owner_id ON images (owner_id)"
                )
            # 舊任務補上已處理圖片欄位
            cur.execute(
                "SHOW COLUMNS FROM annotation_tasks "
                "LIKE 'processed_image_ids'"
            )
            processed_image_column = cur.fetchone()

            if processed_image_column is None:
                cur.execute(
                    "ALTER TABLE annotation_tasks "
                    "ADD COLUMN processed_image_ids JSON NULL "
                    "AFTER image_ids"
                )

            cur.execute(
                """
                UPDATE annotation_tasks
                SET processed_image_ids = CASE
                    WHEN status = 'completed' THEN image_ids
                    ELSE JSON_ARRAY()
                END
                WHERE processed_image_ids IS NULL
                """
            )

            if (
                processed_image_column is None
                or processed_image_column["Null"] == "YES"
            ):
                cur.execute(
                    "ALTER TABLE annotation_tasks "
                    "MODIFY COLUMN processed_image_ids JSON NOT NULL"
                )

            # 舊任務補上資料集與通知版本欄位
            task_version_migrations = [
                (
                    "dataset_version",
                    "INT NOT NULL DEFAULT 0 "
                    "AFTER processed_image_ids",
                ),
                (
                    "notified_dataset_version",
                    "INT NOT NULL DEFAULT 0 "
                    "AFTER dataset_version",
                ),
            ]

            for column, ddl in task_version_migrations:
                cur.execute(
                    "SHOW COLUMNS FROM annotation_tasks "
                    f"LIKE '{column}'"
                )

                if cur.fetchone() is None:
                    cur.execute(
                        "ALTER TABLE annotation_tasks "
                        f"ADD COLUMN {column} {ddl}"
                    )

            # 舊的已完成 ZIP 視為第 1 版
            cur.execute(
                """
                UPDATE annotation_tasks
                SET dataset_version = 1
                WHERE status = 'completed'
                  AND dataset_zip_path <> ''
                  AND dataset_version = 0
                """
            )
            # 確保 parameters 資料表結構正確
            cur.execute("SHOW COLUMNS FROM parameters LIKE 'key'")
            if cur.fetchone() is None:
                cur.execute("DROP TABLE IF EXISTS parameters")
                cur.execute("""
                    CREATE TABLE parameters (
                        `key` VARCHAR(64) PRIMARY KEY,
                        value FLOAT NOT NULL DEFAULT 0
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)

    # ---------- 影像 ----------
    def add_image(self, img: ImageRecord) -> ImageRecord:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO images "
                "(id, owner_id, filename, path, width, height, file_hash, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    img.id, img.owner_id, img.filename, img.path,
                    img.width, img.height, img.file_hash, img.created_at,
                ),
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

    def delete_images_batch(self, image_ids: list[str]) -> list[str]:
        """批次刪除多張圖，單一 Transaction 完成。"""
        if not image_ids:
            return []
        with self._tx() as cur:
            fmt = ",".join(["%s"] * len(image_ids))
            cur.execute(f"SELECT path FROM images WHERE id IN ({fmt}) FOR UPDATE", tuple(image_ids))
            paths = [r["path"] for r in cur.fetchall() if r.get("path")]
            cur.execute(f"SELECT mask_path FROM segments WHERE image_id IN ({fmt})", tuple(image_ids))
            paths += [r["mask_path"] for r in cur.fetchall() if r.get("mask_path")]
            cur.execute(f"DELETE FROM segments WHERE image_id IN ({fmt})", tuple(image_ids))
            cur.execute(f"DELETE FROM images WHERE id IN ({fmt})", tuple(image_ids))
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

    def delete_segments_batch(self, seg_ids: list[str]) -> list[str]:
        """批次刪除多個遮罩片段，單一 Transaction 完成。"""
        if not seg_ids:
            return []
        with self._tx() as cur:
            fmt = ",".join(["%s"] * len(seg_ids))
            cur.execute(f"SELECT mask_path FROM segments WHERE id IN ({fmt}) FOR UPDATE", tuple(seg_ids))
            paths = [r["mask_path"] for r in cur.fetchall() if r.get("mask_path")]
            cur.execute(f"DELETE FROM segments WHERE id IN ({fmt})", tuple(seg_ids))
        return [p for p in paths if p]

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

    # ---------- 標註任務 ----------
    def add_task(self, task: AnnotationTask, ) -> AnnotationTask:
        with self._tx() as cursor:
            cursor.execute(
                """
                INSERT INTO annotation_tasks (
                    id,
                    user_id,
                    line_user_id,
                    prompt,
                    image_ids,
                    processed_image_ids,
                    dataset_version,
                    notified_dataset_version,
                    status,
                    dataset_zip_path,
                    best_model_path,
                    download_token,
                    error_message,
                    created_at,
                    updated_at
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                )
                """,
                (
                    task.id,
                    task.user_id,
                    task.line_user_id,
                    task.prompt,
                    json.dumps(task.image_ids),
                    json.dumps(task.processed_image_ids),
                    task.dataset_version,
                    task.notified_dataset_version,
                    task.status,
                    task.dataset_zip_path,
                    task.best_model_path,
                    task.download_token,
                    task.error_message,
                    task.created_at,
                    task.updated_at,
                ),
            )

        return task

    def get_task(
        self,
        task_id: str,
    ) -> AnnotationTask | None:
        with self._tx() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM annotation_tasks
                WHERE id = %s
                LIMIT 1
                """,
                (task_id,),
            )

            row = cursor.fetchone()

        return _row_to_task(row) if row else None

    def list_tasks_by_line_user_id(
        self,
        line_user_id: str,
    ) -> list[AnnotationTask]:
        """依更新時間由新到舊取得 LINE 使用者的任務。"""
        with self._tx() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM annotation_tasks
                WHERE line_user_id = %s
                ORDER BY updated_at DESC
                """,
                (line_user_id,),
            )

            rows = cursor.fetchall()

        return [
            _row_to_task(row)
            for row in rows
        ]

    def claim_next_pending_task(
        self,
    ) -> AnnotationTask | None:
        updated_at = time.time()

        with self._tx() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM annotation_tasks
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """
            )

            row = cursor.fetchone()

            if row is None:
                return None

            cursor.execute(
                """
                UPDATE annotation_tasks
                SET
                    status = %s,
                    error_message = '',
                    updated_at = %s
                WHERE id = %s
                """,
                (
                    "processing",
                    updated_at,
                    row["id"],
                ),
            )

            row["status"] = "processing"
            row["error_message"] = ""
            row["updated_at"] = updated_at

        return _row_to_task(row)

    def update_task(
        self,
        task: AnnotationTask,
    ) -> AnnotationTask:
        task.updated_at = time.time()

        with self._tx() as cursor:
            cursor.execute(
                """
                UPDATE annotation_tasks
                SET
                    user_id = %s,
                    line_user_id = %s,
                    prompt = %s,
                    image_ids = %s,
                    processed_image_ids = %s,
                    dataset_version = %s,
                    notified_dataset_version = %s,
                    status = %s,
                    dataset_zip_path = %s,
                    best_model_path = %s,
                    download_token = %s,
                    error_message = %s,
                    updated_at = %s
                WHERE id = %s
                """,
                (
                    task.user_id,
                    task.line_user_id,
                    task.prompt,
                    json.dumps(task.image_ids),
                    json.dumps(task.processed_image_ids),
                    task.dataset_version,
                    task.notified_dataset_version,
                    task.status,
                    task.dataset_zip_path,
                    task.best_model_path,
                    task.download_token,
                    task.error_message,
                    task.updated_at,
                    task.id,
                ),
            )

        return task

    def assign_tasks_to_user(
        self,
        line_user_id: str,
        user_id: str,
    ) -> int:
        updated_at = time.time()

        with self._tx() as cursor:
            cursor.execute(
                """
                SELECT image_ids
                FROM annotation_tasks
                WHERE line_user_id = %s
                  AND (user_id = '' OR user_id = %s)
                FOR UPDATE
                """,
                (line_user_id, user_id),
            )
            image_ids: set[str] = set()
            for row in cursor.fetchall():
                raw_ids = row["image_ids"]
                if isinstance(raw_ids, str):
                    raw_ids = json.loads(raw_ids)
                image_ids.update(str(image_id) for image_id in raw_ids)

            assigned_count = cursor.execute(
                """
                UPDATE annotation_tasks
                SET
                    user_id = %s,
                    updated_at = %s
                WHERE line_user_id = %s
                  AND user_id = ''
                """,
                (
                    user_id,
                    updated_at,
                    line_user_id,
                ),
            )
            if image_ids:
                placeholders = ", ".join(["%s"] * len(image_ids))
                cursor.execute(
                    f"UPDATE images SET owner_id = %s "
                    f"WHERE owner_id = '' AND id IN ({placeholders})",
                    (user_id, *sorted(image_ids)),
                )

        return assigned_count

    # ---------- 使用者 / 登入 ----------
    def _write_user(self, cur, user: User) -> None:
        cur.execute(
            "REPLACE INTO users (id, username, password_hash, email, email_verified,"
            " otp_hash, otp_expires, otp_attempts, role, created_at,"
            " line_user_id, display_name, avatar_url)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (user.id, user.username, user.password_hash, user.email,
             int(user.email_verified), user.otp_hash, user.otp_expires,
             user.otp_attempts, user.role,
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

    # ---------- 參數設定 ----------
    def get_parameters(self) -> dict[str, float]:
        with self._tx() as cur:
            cur.execute("SELECT `key`, value FROM parameters")
            rows = cur.fetchall()
        return {r["key"]: float(r["value"]) for r in rows}

    def set_parameter(self, key: str, value: float) -> None:
        with self._tx() as cur:
            cur.execute("REPLACE INTO parameters (`key`, value) VALUES (%s, %s)", (key, value))

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
