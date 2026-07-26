from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from app import create_app
from app.services.task_processor import process_task


def main() -> None:
    app = create_app()
    repo = app.repo
    pipeline = app.pipeline
    cfg = app.smart_config

    task = repo.claim_next_pending_task()

    if task is None:
        print("目前沒有 pending 任務。")
        return

    print(
        f"開始處理任務：{task.id}"
    )

    try:
        updated_task = process_task(
            repo,
            pipeline,
            task,
            cfg.data_dir / "tasks",
        )
    except Exception as exc:
        task.status = "failed"
        task.error_message = str(exc)
        repo.update_task(task)

        print(f"任務失敗：{task.id}")
        print(f"錯誤原因：{exc}")
        raise

    print(f"任務完成：{updated_task.id}")
    print(f"ZIP 路徑：{updated_task.dataset_zip_path}")

if __name__ == "__main__":
    main()