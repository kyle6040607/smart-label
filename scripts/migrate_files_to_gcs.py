"""把資料庫引用的既有本機檔案搬到 Google Cloud Storage。

請在舊檔案仍存在的機器上執行：

    uv run python scripts/migrate_files_to_gcs.py --dry-run
    uv run python scripts/migrate_files_to_gcs.py

上傳成功後才會以舊路徑作為條件更新 DB。物件名稱固定、已是 gs:// 的
紀錄會跳過，因此中斷後可安全重跑。腳本不會刪除任何本機來源檔案。
"""
from __future__ import annotations

import argparse
import mimetypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import config
from app.repository import Repository
from app.repository_mysql import MySQLRepository
from app.storage import StorageService, build_storage


Kind = Literal["image", "segment", "task"]


@dataclass(frozen=True)
class MigrationItem:
    kind: Kind
    record_id: str
    old_reference: str
    object_name: str
    content_type: str
    image_id: str = ""


@dataclass
class MigrationResult:
    migrated: int = 0
    skipped: int = 0
    failed: int = 0
    orphaned: int = 0


def _content_type(path: str, fallback: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or fallback


def _image_object_name(image) -> str:
    suffix = Path(image.path).suffix or Path(image.filename).suffix
    return f"images/{image.id}{suffix.lower()}"


def _json_items(repo: Repository) -> Iterator[MigrationItem]:
    for image in repo.list_images():
        if not image.path:
            continue
        yield MigrationItem(
            kind="image",
            record_id=image.id,
            old_reference=image.path,
            object_name=_image_object_name(image),
            content_type=_content_type(
                image.path or image.filename,
                "application/octet-stream",
            ),
        )

    for segment in repo.list_segments():
        if not segment.mask_path:
            continue
        yield MigrationItem(
            kind="segment",
            record_id=segment.id,
            image_id=segment.image_id,
            old_reference=segment.mask_path,
            object_name=f"masks/{segment.image_id}_{segment.id}.png",
            content_type="image/png",
        )

    for task in repo.tasks.values():
        if not task.dataset_zip_path:
            continue
        yield MigrationItem(
            kind="task",
            record_id=task.id,
            old_reference=task.dataset_zip_path,
            object_name=(
                f"datasets/{task.id}/"
                f"dataset_v{max(task.dataset_version, 1)}.zip"
            ),
            content_type="application/zip",
        )


def _mysql_pages(
    repo: MySQLRepository,
    query: str,
    batch_size: int,
) -> Iterator[list[dict]]:
    last_id = ""
    while True:
        with repo._tx() as cursor:  # noqa: SLF001 - paged migration scan
            cursor.execute(query, (last_id, batch_size))
            rows = cursor.fetchall()
        if not rows:
            return
        yield rows
        last_id = rows[-1]["id"]


def _mysql_items(
    repo: MySQLRepository,
    batch_size: int,
) -> Iterator[MigrationItem]:
    image_query = (
        "SELECT id, filename, path FROM images "
        "WHERE path <> '' AND id > %s ORDER BY id LIMIT %s"
    )
    for rows in _mysql_pages(repo, image_query, batch_size):
        for row in rows:
            suffix = (
                Path(row["path"]).suffix
                or Path(row["filename"]).suffix
            ).lower()
            yield MigrationItem(
                kind="image",
                record_id=row["id"],
                old_reference=row["path"],
                object_name=f"images/{row['id']}{suffix}",
                content_type=_content_type(
                    row["path"] or row["filename"],
                    "application/octet-stream",
                ),
            )

    segment_query = (
        "SELECT id, image_id, mask_path FROM segments "
        "WHERE mask_path <> '' AND id > %s ORDER BY id LIMIT %s"
    )
    for rows in _mysql_pages(repo, segment_query, batch_size):
        for row in rows:
            yield MigrationItem(
                kind="segment",
                record_id=row["id"],
                image_id=row["image_id"],
                old_reference=row["mask_path"],
                object_name=(
                    f"masks/{row['image_id']}_{row['id']}.png"
                ),
                content_type="image/png",
            )

    task_query = (
        "SELECT id, dataset_version, dataset_zip_path "
        "FROM annotation_tasks "
        "WHERE dataset_zip_path <> '' AND id > %s "
        "ORDER BY id LIMIT %s"
    )
    for rows in _mysql_pages(repo, task_query, batch_size):
        for row in rows:
            yield MigrationItem(
                kind="task",
                record_id=row["id"],
                old_reference=row["dataset_zip_path"],
                object_name=(
                    f"datasets/{row['id']}/"
                    f"dataset_v"
                    f"{max(int(row['dataset_version']), 1)}.zip"
                ),
                content_type="application/zip",
            )


def _items(
    repo: Repository | MySQLRepository,
    batch_size: int,
) -> Iterator[MigrationItem]:
    if isinstance(repo, MySQLRepository):
        yield from _mysql_items(repo, batch_size)
        return
    yield from _json_items(repo)


def _resolve_source(
    item: MigrationItem,
    *,
    base_dir: Path,
    source_root: Path | None,
) -> Path:
    stored_path = Path(item.old_reference)
    candidates = [stored_path]

    if not stored_path.is_absolute():
        candidates.append(base_dir / stored_path)
        candidates.append(Path.cwd() / stored_path)

    if source_root is not None:
        candidates.append(source_root / stored_path.name)
        if item.kind == "image":
            candidates.append(source_root / "uploads" / stored_path.name)
        elif item.kind == "segment":
            candidates.append(source_root / "masks" / stored_path.name)
        else:
            candidates.append(
                source_root
                / "tasks"
                / item.record_id
                / stored_path.name
            )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return candidates[0]


def _update_json(
    repo: Repository,
    item: MigrationItem,
    new_reference: str,
) -> bool:
    if item.kind == "image":
        record = repo.get_image(item.record_id)
        if record is None or record.path != item.old_reference:
            return False
        record.path = new_reference
        repo.add_image(record)
        return True

    if item.kind == "segment":
        record = repo.get_segment(item.record_id)
        if record is None or record.mask_path != item.old_reference:
            return False
        record.mask_path = new_reference
        repo.update_segment(record)
        return True

    record = repo.get_task(item.record_id)
    if record is None or record.dataset_zip_path != item.old_reference:
        return False
    record.dataset_zip_path = new_reference
    repo.update_task(record)
    return True


def _update_mysql(
    repo: MySQLRepository,
    item: MigrationItem,
    new_reference: str,
) -> bool:
    statements = {
        "image": (
            "UPDATE images SET path=%s WHERE id=%s AND path=%s"
        ),
        "segment": (
            "UPDATE segments SET mask_path=%s "
            "WHERE id=%s AND mask_path=%s"
        ),
        "task": (
            "UPDATE annotation_tasks SET dataset_zip_path=%s "
            "WHERE id=%s AND dataset_zip_path=%s"
        ),
    }
    with repo._tx() as cursor:  # noqa: SLF001 - compare-and-swap migration update
        updated = cursor.execute(
            statements[item.kind],
            (new_reference, item.record_id, item.old_reference),
        )
    return updated == 1


def _update_reference(
    repo: Repository | MySQLRepository,
    item: MigrationItem,
    new_reference: str,
) -> bool:
    if isinstance(repo, MySQLRepository):
        return _update_mysql(repo, item, new_reference)
    return _update_json(repo, item, new_reference)


def migrate(
    repo: Repository | MySQLRepository,
    storage: StorageService | None,
    *,
    base_dir: Path,
    source_root: Path | None = None,
    dry_run: bool = False,
    batch_size: int = 500,
) -> MigrationResult:
    if batch_size <= 0:
        raise ValueError("batch_size 必須大於 0")
    if not dry_run and (
        storage is None or storage.backend_name != "gcs"
    ):
        raise ValueError("正式遷移必須使用已啟用的 GCS StorageService")

    result = MigrationResult()

    for item in _items(repo, batch_size):
        if item.old_reference.startswith("gs://"):
            result.skipped += 1
            continue

        source = _resolve_source(
            item,
            base_dir=base_dir,
            source_root=source_root,
        )
        description = (
            f"{item.kind}:{item.record_id} "
            f"{source} -> gs://.../{item.object_name}"
        )

        if not source.is_file():
            result.failed += 1
            print(f"[失敗] 找不到來源檔案：{description}", file=sys.stderr)
            continue

        if dry_run:
            result.migrated += 1
            print(f"[預覽] {description}")
            continue

        new_reference = ""
        try:
            assert storage is not None
            new_reference = storage.save_file(
                item.object_name,
                source,
                item.content_type,
            )
            if not _update_reference(repo, item, new_reference):
                raise RuntimeError("DB 路徑已被其他程序變更")
        except Exception as exc:
            result.failed += 1
            if new_reference:
                result.orphaned += 1
                print(
                    f"[孤兒物件] {new_reference}",
                    file=sys.stderr,
                )
            print(f"[失敗] {description}：{exc}", file=sys.stderr)
            continue

        result.migrated += 1
        print(f"[完成] {item.old_reference} -> {new_reference}")

    return result


def count_local_references(
    repo: Repository | MySQLRepository,
) -> int:
    if isinstance(repo, MySQLRepository):
        statements = (
            "SELECT COUNT(*) AS count FROM images "
            "WHERE path <> '' AND path NOT LIKE 'gs://%'",
            "SELECT COUNT(*) AS count FROM segments "
            "WHERE mask_path <> '' AND mask_path NOT LIKE 'gs://%'",
            "SELECT COUNT(*) AS count FROM annotation_tasks "
            "WHERE dataset_zip_path <> '' "
            "AND dataset_zip_path NOT LIKE 'gs://%'",
        )
        total = 0
        with repo._tx() as cursor:  # noqa: SLF001 - migration verification
            for statement in statements:
                cursor.execute(statement)
                total += int(cursor.fetchone()["count"])
        return total

    return sum(
        not reference.startswith("gs://")
        for reference in (
            *(
                image.path
                for image in repo.list_images()
                if image.path
            ),
            *(
                segment.mask_path
                for segment in repo.list_segments()
                if segment.mask_path
            ),
            *(
                task.dataset_zip_path
                for task in repo.tasks.values()
                if task.dataset_zip_path
            ),
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 images、segments、annotation_tasks 的本機檔案搬到 GCS",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只檢查來源路徑與顯示映射，不上傳、不更新 DB",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只執行完整來源檔案檢查；等同不寫入的預檢模式",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="MySQL 每次讀取的紀錄數（預設 500）",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "舊 data 目錄的新位置；當 DB 內路徑已無法直接解析時，"
            "會嘗試其 uploads、masks、tasks 子目錄"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not config.use_gcs:
        raise SystemExit("請先設定 USE_GCS=1")
    if not config.gcs_bucket_name:
        raise SystemExit("請先設定 GCS_BUCKET_NAME 或 GCS_BUCKET")

    repo: Repository | MySQLRepository
    if config.use_mysql:
        repo = MySQLRepository(config)
    else:
        if not config.db_file.exists():
            raise SystemExit(f"找不到 JSON 資料檔：{config.db_file}")
        repo = Repository(config.db_file)

    if args.batch_size <= 0:
        raise SystemExit("--batch-size 必須大於 0")

    preflight = migrate(
        repo,
        None,
        base_dir=config.base_dir,
        source_root=args.source_root,
        dry_run=True,
        batch_size=args.batch_size,
    )
    print(
        "預檢結果："
        f"ready={preflight.migrated} "
        f"skipped={preflight.skipped} "
        f"failed={preflight.failed}"
    )
    if preflight.failed:
        raise SystemExit(1)
    if args.dry_run or args.preflight_only:
        return

    result = migrate(
        repo,
        build_storage(config),
        base_dir=config.base_dir,
        source_root=args.source_root,
        batch_size=args.batch_size,
    )
    remaining = count_local_references(repo)
    print(
        "遷移結果："
        f"migrated={result.migrated} "
        f"skipped={result.skipped} "
        f"failed={result.failed} "
        f"orphaned={result.orphaned} "
        f"remaining_local={remaining}"
    )

    if result.failed or remaining:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
