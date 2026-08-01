"""LIFF 背景任務 Worker。

loop 模式會常駐輪詢，適合 Cloud Run Worker Pool；drain 模式會在佇列清空後
離開，適合 Cloud Run Job 或其他批次執行環境。
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app
from app.services.task_runner import TaskRunResult, run_next_task


@dataclass(frozen=True)
class WorkerStats:
    completed: int = 0
    failed: int = 0


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必須是大於 0 的數字")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="執行 LIFF 背景任務")
    parser.add_argument(
        "--mode",
        choices=("loop", "drain"),
        default="loop",
        help="loop 常駐輪詢；drain 清空目前佇列後離開",
    )
    parser.add_argument(
        "--poll-seconds",
        type=_positive_float,
        default=_positive_float(
            os.getenv("TASK_WORKER_POLL_SECONDS", "3")
        ),
        help="loop 模式沒有任務時的輪詢間隔",
    )
    return parser.parse_args(argv)


def run_worker(
    app,
    *,
    mode: str,
    poll_seconds: float,
    stop_event: threading.Event | None = None,
    run_task: Callable[[object], TaskRunResult] = run_next_task,
) -> WorkerStats:
    """持續處理任務；每次任務結束後才響應停止事件。"""
    if mode not in {"loop", "drain"}:
        raise ValueError(f"不支援的 Worker 模式：{mode}")

    stop_event = stop_event or threading.Event()
    completed = 0
    failed = 0

    while not stop_event.is_set():
        result = run_task(app)

        if result == TaskRunResult.COMPLETED:
            completed += 1
            continue

        if result == TaskRunResult.FAILED:
            failed += 1
            continue

        if result != TaskRunResult.IDLE:
            raise ValueError(f"未知的任務執行結果：{result}")

        if mode == "drain":
            break

        stop_event.wait(poll_seconds)

    return WorkerStats(completed=completed, failed=failed)


def _install_signal_handlers(
    stop_event: threading.Event,
    logger,
) -> None:
    def request_shutdown(signum, _frame) -> None:
        logger.info(
            "收到關機訊號 %s，將在目前任務完成後停止 Worker",
            signum,
        )
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = create_app()

    if args.mode == "loop" and not app.smart_config.use_mysql:
        app.logger.error(
            "loop 模式需要 MySQL；JSON Repository 無法安全地由 Web 與 Worker 跨程序共用"
        )
        return 2

    stop_event = threading.Event()
    _install_signal_handlers(stop_event, app.logger)

    app.logger.info(
        "啟動 LIFF Worker：mode=%s poll_seconds=%s",
        args.mode,
        args.poll_seconds,
    )

    # App context 與其資源會在 Worker 結束時統一釋放；單一任務內的 ZIP writer、
    # storage reader 等 I/O 仍由各 service 自己的 with 區塊即時關閉。
    with app.app_context():
        stats = run_worker(
            app,
            mode=args.mode,
            poll_seconds=args.poll_seconds,
            stop_event=stop_event,
        )

    app.logger.info(
        "LIFF Worker 已停止：completed=%s failed=%s",
        stats.completed,
        stats.failed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
