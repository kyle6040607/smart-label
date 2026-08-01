from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app
from app.services.task_runner import TaskRunResult, run_next_task


def main() -> None:
    app = create_app()
    result = run_next_task(app)

    if result == TaskRunResult.IDLE:
        print("目前沒有 pending 任務。")
        return

    if result == TaskRunResult.FAILED:
        print("任務處理失敗，詳細原因請查看 log。")
        raise SystemExit(1)

    print("任務處理完成。")


if __name__ == "__main__":
    main()
