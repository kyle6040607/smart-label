"""把既有的 data/store.json 資料搬進 MySQL（含 Cloud SQL）。

用法：先在 .env（或環境變數）設好 MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD /
MYSQL_DATABASE（Cloud Run 以外的環境用 TCP 即可），然後：

    uv run python scripts/migrate_json_to_mysql.py

已存在的同 id 資料會被覆蓋（REPLACE / 先刪後插），重跑安全。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import config
from app.repository import Repository
from app.repository_mysql import MySQLRepository


def main() -> None:
    if not config.use_mysql:
        raise SystemExit(
            "MySQL 連線未設定：請先在 .env 設 MYSQL_HOST（或 MYSQL_UNIX_SOCKET）"
            "、MYSQL_USER、MYSQL_PASSWORD、MYSQL_DATABASE"
        )
    if not config.db_file.exists():
        raise SystemExit(f"找不到 JSON 資料檔：{config.db_file}")

    src = Repository(config.db_file)
    dst = MySQLRepository(config)

    with dst._tx() as cur:  # noqa: SLF001 — 遷移腳本直接用同一個交易批次寫入
        for img in src.images.values():
            cur.execute(
                "REPLACE INTO images (id, filename, path, width, height, file_hash, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (img.id, img.filename, img.path, img.width, img.height,
                 img.file_hash, img.created_at),
            )
        for seg in src.segments.values():
            dst._write_segment(cur, seg)  # noqa: SLF001
        for user in src.users.values():
            dst._write_user(cur, user)  # noqa: SLF001
        for ex in src.examples.values():
            cur.execute(
                "REPLACE INTO examples (id, label, feature, source_segment_id, created_at)"
                " VALUES (%s, %s, %s, %s, %s)",
                (ex.id, ex.label, json.dumps(ex.feature),
                 ex.source_segment_id, ex.created_at),
            )
        for task in src.tasks.values():
            cur.execute(
                """
                REPLACE INTO annotation_tasks (
                    id, user_id, line_user_id, prompt, image_ids, status,
                    dataset_zip_path, best_model_path, download_token,
                    error_message, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    task.id, task.user_id, task.line_user_id, task.prompt,
                    json.dumps(task.image_ids), task.status,
                    task.dataset_zip_path, task.best_model_path,
                    task.download_token, task.error_message,
                    task.created_at, task.updated_at,
                ),
            )
        for s in src.line_sessions.values():
            dst._write_session(cur, s)  # noqa: SLF001
        for key, value in src.parameters.items():
            cur.execute(
                "REPLACE INTO parameters (`key`, value) VALUES (%s, %s)",
                (key, value),
            )

    print(
        f"遷移完成：images={len(src.images)} segments={len(src.segments)}"
        f" examples={len(src.examples)} users={len(src.users)}"
        f" tasks={len(src.tasks)} line_sessions={len(src.line_sessions)}"
        f" parameters={len(src.parameters)}"
    )


if __name__ == "__main__":
    main()
