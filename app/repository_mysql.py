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
import uuid
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
    detection_confidence FLOAT NOT NULL DEFAULT 0,
    needs_review TINYINT(1) NOT NULL DEFAULT 0,
    annotation_task_id VARCHAR(32) NOT NULL DEFAULT '',
    task_attempt_token VARCHAR(64) NOT NULL DEFAULT '',
    human_label VARCHAR(64) NULL,
    reviewed TINYINT(1) NOT NULL DEFAULT 0,
    INDEX idx_segments_image_id (image_id),
    INDEX idx_segments_task_attempt (annotation_task_id, task_attempt_token)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS examples (
    id VARCHAR(32) PRIMARY KEY,
    owner_id VARCHAR(32) NOT NULL DEFAULT '',
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
    claimed_by VARCHAR(128) NOT NULL DEFAULT '',
    claim_token VARCHAR(64) NOT NULL DEFAULT '',
    processing_started_at DOUBLE NOT NULL DEFAULT 0,
    heartbeat_at DOUBLE NOT NULL DEFAULT 0,
    lease_expires_at DOUBLE NOT NULL DEFAULT 0,
    attempt_count INT NOT NULL DEFAULT 0,
    next_attempt_at DOUBLE NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL,
    settings_snapshot JSON NOT NULL,
    segment_count INT NOT NULL DEFAULT 0,
    exported_count INT NOT NULL DEFAULT 0,
    excluded_count INT NOT NULL DEFAULT 0,
    no_detection_image_ids JSON NOT NULL,
    excluded_results JSON NOT NULL,
    completion_reason VARCHAR(64) NOT NULL DEFAULT '',
    dataset_zip_path VARCHAR(512) NOT NULL DEFAULT '',
    best_model_path VARCHAR(512) NOT NULL DEFAULT '',
    download_token VARCHAR(64) NOT NULL,
    error_message TEXT NOT NULL,
    failure_notified_at DOUBLE NOT NULL DEFAULT 0,
    created_at DOUBLE NOT NULL,
    updated_at DOUBLE NOT NULL,
    INDEX idx_annotation_tasks_user_id (user_id),
    INDEX idx_annotation_tasks_line_user_id (line_user_id),
    INDEX idx_annotation_tasks_claimable (status, next_attempt_at, created_at),
    INDEX idx_annotation_tasks_lease (status, lease_expires_at),
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
        detection_confidence=float(r.get("detection_confidence", 0.0)),
        needs_review=bool(r["needs_review"]),
        annotation_task_id=r.get("annotation_task_id", ""),
        task_attempt_token=r.get("task_attempt_token", ""),
        human_label=r["human_label"], reviewed=bool(r["reviewed"]),
    )


def _row_to_example(r: dict) -> LabelExample:
    return LabelExample(
        id=r["id"], owner_id=r.get("owner_id", ""),
        label=r["label"], feature=json.loads(r["feature"]),
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
        claimed_by=row.get("claimed_by", ""),
        claim_token=row.get("claim_token", ""),
        processing_started_at=float(row.get("processing_started_at", 0.0)),
        heartbeat_at=float(row.get("heartbeat_at", 0.0)),
        lease_expires_at=float(row.get("lease_expires_at", 0.0)),
        attempt_count=int(row.get("attempt_count", 0)),
        next_attempt_at=float(row.get("next_attempt_at", 0.0)),
        last_error=row.get("last_error", ""),
        settings_snapshot=json.loads(row.get("settings_snapshot") or "{}"),
        segment_count=int(row.get("segment_count", 0)),
        exported_count=int(row.get("exported_count", 0)),
        excluded_count=int(row.get("excluded_count", 0)),
        no_detection_image_ids=json.loads(
            row.get("no_detection_image_ids") or "[]"
        ),
        excluded_results=json.loads(row.get("excluded_results") or "[]"),
        completion_reason=row.get("completion_reason", ""),
        dataset_zip_path=row["dataset_zip_path"],
        best_model_path=row["best_model_path"],
        download_token=row["download_token"],
        error_message=row["error_message"],
        failure_notified_at=float(row.get("failure_notified_at", 0.0)),
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

            cur.execute("SHOW COLUMNS FROM examples LIKE 'owner_id'")
            if cur.fetchone() is None:
                cur.execute(
                    "ALTER TABLE examples "
                    "ADD COLUMN owner_id VARCHAR(32) NOT NULL DEFAULT '' "
                    "AFTER id"
                )
                cur.execute(
                    "CREATE INDEX idx_examples_owner_id ON examples (owner_id)"
                )
                # 可從來源片段追溯的舊範例，自動繼承圖片擁有者。
                cur.execute(
                    """
                    UPDATE examples AS example
                    JOIN segments AS segment
                      ON segment.id = example.source_segment_id
                    JOIN images AS image
                      ON image.id = segment.image_id
                    SET example.owner_id = image.owner_id
                    WHERE example.owner_id = ''
                    """
                )

            # LIFF Segment 的偵測信心與 attempt 歸屬。
            segment_migrations = [
                (
                    "detection_confidence",
                    "FLOAT NOT NULL DEFAULT 0 AFTER confidence",
                ),
                (
                    "annotation_task_id",
                    "VARCHAR(32) NOT NULL DEFAULT '' AFTER needs_review",
                ),
                (
                    "task_attempt_token",
                    "VARCHAR(64) NOT NULL DEFAULT '' AFTER annotation_task_id",
                ),
            ]
            for column, ddl in segment_migrations:
                cur.execute(f"SHOW COLUMNS FROM segments LIKE '{column}'")
                if cur.fetchone() is None:
                    cur.execute(
                        f"ALTER TABLE segments ADD COLUMN {column} {ddl}"
                    )

            cur.execute(
                "SHOW INDEX FROM segments "
                "WHERE Key_name = 'idx_segments_task_attempt'"
            )
            if cur.fetchone() is None:
                cur.execute(
                    "CREATE INDEX idx_segments_task_attempt "
                    "ON segments (annotation_task_id, task_attempt_token)"
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

            # Worker lease/retry、結果統計與低信心縮圖欄位。JSON 欄位先允許
            # NULL、補值後再收緊，避免既有 annotation_tasks 遷移失敗。
            task_scalar_migrations = [
                ("claimed_by", "VARCHAR(128) NOT NULL DEFAULT '' AFTER status"),
                ("claim_token", "VARCHAR(64) NOT NULL DEFAULT '' AFTER claimed_by"),
                ("processing_started_at", "DOUBLE NOT NULL DEFAULT 0 AFTER claim_token"),
                ("heartbeat_at", "DOUBLE NOT NULL DEFAULT 0 AFTER processing_started_at"),
                ("lease_expires_at", "DOUBLE NOT NULL DEFAULT 0 AFTER heartbeat_at"),
                ("attempt_count", "INT NOT NULL DEFAULT 0 AFTER lease_expires_at"),
                ("next_attempt_at", "DOUBLE NOT NULL DEFAULT 0 AFTER attempt_count"),
                ("last_error", "TEXT NULL AFTER next_attempt_at"),
                ("segment_count", "INT NOT NULL DEFAULT 0"),
                ("exported_count", "INT NOT NULL DEFAULT 0 AFTER segment_count"),
                ("excluded_count", "INT NOT NULL DEFAULT 0 AFTER exported_count"),
                ("completion_reason", "VARCHAR(64) NOT NULL DEFAULT ''"),
                ("failure_notified_at", "DOUBLE NOT NULL DEFAULT 0 AFTER error_message"),
            ]
            for column, ddl in task_scalar_migrations:
                cur.execute(
                    "SHOW COLUMNS FROM annotation_tasks "
                    f"LIKE '{column}'"
                )
                if cur.fetchone() is None:
                    cur.execute(
                        "ALTER TABLE annotation_tasks "
                        f"ADD COLUMN {column} {ddl}"
                    )

            task_json_migrations = [
                ("settings_snapshot", "{}", "AFTER last_error"),
                ("no_detection_image_ids", "[]", "AFTER excluded_count"),
                ("excluded_results", "[]", "AFTER no_detection_image_ids"),
            ]
            for column, json_default, position in task_json_migrations:
                cur.execute(
                    "SHOW COLUMNS FROM annotation_tasks "
                    f"LIKE '{column}'"
                )
                column_info = cur.fetchone()
                if column_info is None:
                    cur.execute(
                        "ALTER TABLE annotation_tasks "
                        f"ADD COLUMN {column} JSON NULL {position}"
                    )
                cur.execute(
                    f"UPDATE annotation_tasks SET {column}=%s "
                    f"WHERE {column} IS NULL",
                    (json_default,),
                )
                if column_info is None or column_info.get("Null") == "YES":
                    cur.execute(
                        "ALTER TABLE annotation_tasks "
                        f"MODIFY COLUMN {column} JSON NOT NULL"
                    )

            cur.execute(
                "UPDATE annotation_tasks SET last_error='' "
                "WHERE last_error IS NULL"
            )
            cur.execute(
                "SHOW COLUMNS FROM annotation_tasks LIKE 'last_error'"
            )
            last_error_info = cur.fetchone()
            if last_error_info and last_error_info.get("Null") == "YES":
                cur.execute(
                    "ALTER TABLE annotation_tasks "
                    "MODIFY COLUMN last_error TEXT NOT NULL"
                )

            # 讓升級前已在 processing 的任務也能在 lease 到期後被回收。
            cur.execute(
                """
                UPDATE annotation_tasks
                SET heartbeat_at = CASE
                        WHEN heartbeat_at = 0 THEN updated_at
                        ELSE heartbeat_at
                    END,
                    lease_expires_at = CASE
                        WHEN lease_expires_at = 0 THEN updated_at + %s
                        ELSE lease_expires_at
                    END
                WHERE status = 'processing'
                """,
                (self._cfg.task_lease_seconds,),
            )

            for index_name, columns in (
                (
                    "idx_annotation_tasks_claimable",
                    "status, next_attempt_at, created_at",
                ),
                (
                    "idx_annotation_tasks_lease",
                    "status, lease_expires_at",
                ),
            ):
                cur.execute(
                    "SHOW INDEX FROM annotation_tasks WHERE Key_name=%s",
                    (index_name,),
                )
                if cur.fetchone() is None:
                    cur.execute(
                        f"CREATE INDEX {index_name} "
                        f"ON annotation_tasks ({columns})"
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

    def set_image_hash(self, image_id: str, file_hash: str) -> None:
        """補寫舊資料缺少的 file_hash；只在欄位仍為空時寫入。"""
        with self._tx() as cur:
            cur.execute(
                "UPDATE images SET file_hash=%s WHERE id=%s AND file_hash=''",
                (file_hash, image_id),
            )

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
            " predicted_label, probs, confidence, detection_confidence,"
            " needs_review, annotation_task_id, task_attempt_token,"
            " human_label, reviewed)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (seg.id, seg.image_id, seg.mask_path, json.dumps(list(seg.bbox)),
             seg.area, seg.predicted_label, json.dumps(seg.probs),
             seg.confidence, seg.detection_confidence, seg.needs_review,
             seg.annotation_task_id, seg.task_attempt_token,
             seg.human_label, seg.reviewed),
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
                "INSERT INTO examples "
                "(id, owner_id, label, feature, source_segment_id, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    ex.id, ex.owner_id, ex.label, json.dumps(ex.feature),
                    ex.source_segment_id, ex.created_at,
                ),
            )
        return ex

    def list_examples(self, owner_id: str | None = None) -> list[LabelExample]:
        with self._tx() as cur:
            if owner_id is None:
                cur.execute("SELECT * FROM examples")
            else:
                cur.execute(
                    "SELECT * FROM examples WHERE owner_id=%s",
                    (owner_id,),
                )
            rows = cur.fetchall()
        return [_row_to_example(r) for r in rows]

    def labels(self, owner_id: str | None = None) -> list[str]:
        with self._tx() as cur:
            if owner_id is None:
                cur.execute(
                    "SELECT DISTINCT label FROM examples ORDER BY label"
                )
            else:
                cur.execute(
                    "SELECT DISTINCT label FROM examples "
                    "WHERE owner_id=%s ORDER BY label",
                    (owner_id,),
                )
            rows = cur.fetchall()
        return [r["label"] for r in rows]

    def delete_label(self, label: str, owner_id: str | None = None) -> int:
        """刪掉某類別的所有種子範例，連帶把該類別的人工標記退回送審。"""
        with self._tx() as cur:
            if owner_id is None:
                deleted = cur.execute(
                    "DELETE FROM examples WHERE label=%s",
                    (label,),
                )
                cur.execute(
                    "UPDATE segments SET human_label=NULL, "
                    "reviewed=0, needs_review=1 WHERE human_label=%s",
                    (label,),
                )
            else:
                deleted = cur.execute(
                    "DELETE FROM examples WHERE label=%s AND owner_id=%s",
                    (label, owner_id),
                )
                cur.execute(
                    """
                    UPDATE segments AS segment
                    JOIN images AS image ON image.id = segment.image_id
                    SET segment.human_label=NULL,
                        segment.reviewed=0,
                        segment.needs_review=1
                    WHERE segment.human_label=%s
                      AND image.owner_id=%s
                    """,
                    (label, owner_id),
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
                    claimed_by,
                    claim_token,
                    processing_started_at,
                    heartbeat_at,
                    lease_expires_at,
                    attempt_count,
                    next_attempt_at,
                    last_error,
                    settings_snapshot,
                    segment_count,
                    exported_count,
                    excluded_count,
                    no_detection_image_ids,
                    excluded_results,
                    completion_reason,
                    dataset_zip_path,
                    best_model_path,
                    download_token,
                    error_message,
                    failure_notified_at,
                    created_at,
                    updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s
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
                    task.claimed_by,
                    task.claim_token,
                    task.processing_started_at,
                    task.heartbeat_at,
                    task.lease_expires_at,
                    task.attempt_count,
                    task.next_attempt_at,
                    task.last_error,
                    json.dumps(task.settings_snapshot),
                    task.segment_count,
                    task.exported_count,
                    task.excluded_count,
                    json.dumps(task.no_detection_image_ids),
                    json.dumps(task.excluded_results),
                    task.completion_reason,
                    task.dataset_zip_path,
                    task.best_model_path,
                    task.download_token,
                    task.error_message,
                    task.failure_notified_at,
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
        worker_id: str = "worker",
        lease_seconds: float = 900.0,
        max_attempts: int = 3,
    ) -> AnnotationTask | None:
        now = time.time()
        claim_token = uuid.uuid4().hex

        with self._tx() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM annotation_tasks
                WHERE attempt_count < %s
                  AND (
                    status = 'pending'
                    OR (
                        status = 'retry_wait'
                        AND next_attempt_at <= %s
                    )
                  )
                ORDER BY created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """,
                (max_attempts, now),
            )

            row = cursor.fetchone()

            if row is None:
                return None

            cursor.execute(
                """
                UPDATE annotation_tasks
                SET
                    status = 'processing',
                    claimed_by = %s,
                    claim_token = %s,
                    processing_started_at = %s,
                    heartbeat_at = %s,
                    lease_expires_at = %s,
                    attempt_count = attempt_count + 1,
                    next_attempt_at = 0,
                    error_message = '',
                    updated_at = %s
                WHERE id = %s
                """,
                (
                    worker_id,
                    claim_token,
                    now,
                    now,
                    now + lease_seconds,
                    now,
                    row["id"],
                ),
            )

            row["status"] = "processing"
            row["claimed_by"] = worker_id
            row["claim_token"] = claim_token
            row["processing_started_at"] = now
            row["heartbeat_at"] = now
            row["lease_expires_at"] = now + lease_seconds
            row["attempt_count"] = int(row.get("attempt_count", 0)) + 1
            row["next_attempt_at"] = 0.0
            row["error_message"] = ""
            row["updated_at"] = now

        return _row_to_task(row)

    def heartbeat_task(
        self,
        task_id: str,
        claim_token: str,
        lease_seconds: float,
    ) -> bool:
        now = time.time()
        with self._tx() as cursor:
            updated = cursor.execute(
                """
                UPDATE annotation_tasks
                SET heartbeat_at=%s,
                    lease_expires_at=%s,
                    updated_at=%s
                WHERE id=%s
                  AND status='processing'
                  AND claim_token=%s
                """,
                (
                    now,
                    now + lease_seconds,
                    now,
                    task_id,
                    claim_token,
                ),
            )
        return bool(updated)

    def recover_stale_tasks(
        self,
        *,
        now: float | None = None,
        max_attempts: int = 3,
        limit: int = 10,
    ) -> list[AnnotationTask]:
        """以 SKIP LOCKED 排他回收 lease 到期任務，交易內只改狀態。"""
        now = time.time() if now is None else now
        recovered: list[AnnotationTask] = []
        with self._tx() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM annotation_tasks
                WHERE status='processing'
                  AND lease_expires_at > 0
                  AND lease_expires_at <= %s
                ORDER BY lease_expires_at ASC
                LIMIT %s
                FOR UPDATE SKIP LOCKED
                """,
                (now, limit),
            )
            for row in cursor.fetchall():
                terminal = int(row.get("attempt_count", 0)) >= max_attempts
                status = "failed" if terminal else "retry_wait"
                message = row.get("last_error") or "Worker lease 已逾時"
                next_attempt_at = 0.0 if terminal else now
                cursor.execute(
                    """
                    UPDATE annotation_tasks
                    SET status=%s,
                        claimed_by='',
                        claim_token='',
                        heartbeat_at=0,
                        lease_expires_at=0,
                        next_attempt_at=%s,
                        last_error=%s,
                        error_message=%s,
                        updated_at=%s
                    WHERE id=%s
                    """,
                    (
                        status,
                        next_attempt_at,
                        message,
                        message,
                        now,
                        row["id"],
                    ),
                )
                row.update(
                    status=status,
                    claimed_by="",
                    claim_token="",
                    heartbeat_at=0.0,
                    lease_expires_at=0.0,
                    next_attempt_at=next_attempt_at,
                    last_error=message,
                    error_message=message,
                    updated_at=now,
                )
                recovered.append(_row_to_task(row))
        return recovered

    def fail_or_retry_task(
        self,
        task_id: str,
        claim_token: str,
        error_message: str,
        *,
        max_attempts: int,
        retry_at: float,
    ) -> AnnotationTask | None:
        now = time.time()
        with self._tx() as cursor:
            cursor.execute(
                """
                SELECT * FROM annotation_tasks
                WHERE id=%s AND status='processing' AND claim_token=%s
                LIMIT 1 FOR UPDATE
                """,
                (task_id, claim_token),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            terminal = int(row.get("attempt_count", 0)) >= max_attempts
            status = "failed" if terminal else "retry_wait"
            next_attempt_at = 0.0 if terminal else retry_at
            cursor.execute(
                """
                UPDATE annotation_tasks
                SET status=%s,
                    claimed_by='',
                    claim_token='',
                    heartbeat_at=0,
                    lease_expires_at=0,
                    next_attempt_at=%s,
                    last_error=%s,
                    error_message=%s,
                    updated_at=%s
                WHERE id=%s
                """,
                (
                    status,
                    next_attempt_at,
                    error_message,
                    error_message,
                    now,
                    task_id,
                ),
            )
            row.update(
                status=status,
                claimed_by="",
                claim_token="",
                heartbeat_at=0.0,
                lease_expires_at=0.0,
                next_attempt_at=next_attempt_at,
                last_error=error_message,
                error_message=error_message,
                updated_at=now,
            )
        return _row_to_task(row)

    def complete_claimed_task(
        self,
        task: AnnotationTask,
        claim_token: str,
    ) -> AnnotationTask | None:
        """以 claim_token 作 fencing；過期 Worker 不得發布結果。"""
        now = time.time()
        with self._tx() as cursor:
            updated = cursor.execute(
                """
                UPDATE annotation_tasks
                SET processed_image_ids=%s,
                    dataset_version=%s,
                    status='completed',
                    claimed_by='',
                    claim_token='',
                    heartbeat_at=0,
                    lease_expires_at=0,
                    next_attempt_at=0,
                    last_error='',
                    settings_snapshot=%s,
                    segment_count=%s,
                    exported_count=%s,
                    excluded_count=%s,
                    no_detection_image_ids=%s,
                    excluded_results=%s,
                    completion_reason=%s,
                    dataset_zip_path=%s,
                    error_message='',
                    updated_at=%s
                WHERE id=%s
                  AND status='processing'
                  AND claim_token=%s
                """,
                (
                    json.dumps(task.processed_image_ids),
                    task.dataset_version,
                    json.dumps(task.settings_snapshot),
                    task.segment_count,
                    task.exported_count,
                    task.excluded_count,
                    json.dumps(task.no_detection_image_ids),
                    json.dumps(task.excluded_results),
                    task.completion_reason,
                    task.dataset_zip_path,
                    now,
                    task.id,
                    claim_token,
                ),
            )
        if not updated:
            return None
        task.status = "completed"
        task.claimed_by = ""
        task.claim_token = ""
        task.heartbeat_at = 0.0
        task.lease_expires_at = 0.0
        task.next_attempt_at = 0.0
        task.last_error = ""
        task.error_message = ""
        task.updated_at = now
        return task

    def close_thread_connection(self) -> None:
        """關閉目前 thread 的實體連線；heartbeat thread 結束時呼叫。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

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
                    claimed_by = %s,
                    claim_token = %s,
                    processing_started_at = %s,
                    heartbeat_at = %s,
                    lease_expires_at = %s,
                    attempt_count = %s,
                    next_attempt_at = %s,
                    last_error = %s,
                    settings_snapshot = %s,
                    segment_count = %s,
                    exported_count = %s,
                    excluded_count = %s,
                    no_detection_image_ids = %s,
                    excluded_results = %s,
                    completion_reason = %s,
                    dataset_zip_path = %s,
                    best_model_path = %s,
                    download_token = %s,
                    error_message = %s,
                    failure_notified_at = %s,
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
                    task.claimed_by,
                    task.claim_token,
                    task.processing_started_at,
                    task.heartbeat_at,
                    task.lease_expires_at,
                    task.attempt_count,
                    task.next_attempt_at,
                    task.last_error,
                    json.dumps(task.settings_snapshot),
                    task.segment_count,
                    task.exported_count,
                    task.excluded_count,
                    json.dumps(task.no_detection_image_ids),
                    json.dumps(task.excluded_results),
                    task.completion_reason,
                    task.dataset_zip_path,
                    task.best_model_path,
                    task.download_token,
                    task.error_message,
                    task.failure_notified_at,
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
                cursor.execute(
                    f"""
                    UPDATE examples AS example
                    JOIN segments AS segment
                      ON segment.id = example.source_segment_id
                    SET example.owner_id = %s
                    WHERE example.owner_id = ''
                      AND segment.image_id IN ({placeholders})
                    """,
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

    def transfer_ownership(self, from_user_id: str, to_user_id: str) -> dict[str, int]:
        """把一個帳號名下的圖片、種子範例與任務全部轉移給另一個帳號。

        三張表在同一個交易裡更新，避免只轉移一半就中斷。
        Segment 沒有自己的 owner，擁有權由所屬圖片決定，因此不需處理。
        """
        moved = {"images": 0, "examples": 0, "tasks": 0}
        if not from_user_id or from_user_id == to_user_id:
            return moved

        with self._tx() as cur:
            moved["images"] = cur.execute(
                "UPDATE images SET owner_id=%s WHERE owner_id=%s",
                (to_user_id, from_user_id),
            )
            moved["examples"] = cur.execute(
                "UPDATE examples SET owner_id=%s WHERE owner_id=%s",
                (to_user_id, from_user_id),
            )
            moved["tasks"] = cur.execute(
                "UPDATE annotation_tasks SET user_id=%s, updated_at=%s WHERE user_id=%s",
                (to_user_id, time.time(), from_user_id),
            )

        return moved

    def delete_user(self, user_id: str) -> bool:
        """刪除帳號本身；名下資料請先用 transfer_ownership 轉移，否則會變成無主。"""
        with self._tx() as cur:
            deleted = cur.execute("DELETE FROM users WHERE id=%s", (user_id,))
        return bool(deleted)

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
