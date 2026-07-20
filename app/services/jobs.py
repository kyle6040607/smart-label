"""批量分割的背景執行器。

前端送出一批 image_ids 後立刻拿到 job_id，執行器在背景逐張跑
pipeline，前端輪詢 job 狀態畫進度條，不會被單一 HTTP request 的
timeout 卡住。

刻意只開一個 worker：推論是 CPU-bound，並行只會互相搶核心變更慢，
排隊逐張反而整體最快。

介面預留抽換空間：之後上 GCP 可另寫 CloudTasksJobRunner（submit()
改成對每張圖 enqueue 一個 Cloud Tasks 任務），API 與前端完全不用改。
"""
from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor

from app.models import SegmentJob
from app.repository import Repository
from app.services.pipeline import Pipeline

logger = logging.getLogger(__name__)


class JobRunner:
    def __init__(self, pipeline: Pipeline, repo: Repository):
        self._pipeline = pipeline
        self._repo = repo
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segjob")
        self._futures: dict[str, Future] = {}

    def submit(self, job: SegmentJob) -> None:
        """把 job 排進佇列，立刻返回；實際處理在背景 thread 逐張進行。"""
        future = self._executor.submit(self._run, job.id)
        self._futures[job.id] = future
        # 完成即清（callback 對已完成的 future 會立刻執行，無競態），
        # 長時間執行 _futures 不累積
        future.add_done_callback(lambda _f, jid=job.id: self._futures.pop(jid, None))

    def join(self, timeout: float | None = None) -> None:
        """等所有已送出的 job 跑完（測試用，正式流程走輪詢）。"""
        for future in list(self._futures.values()):
            future.result(timeout=timeout)

    def shutdown(self) -> None:
        """等佇列清空後關閉 executor（測試 fixture / 優雅關機用）。"""
        self._executor.shutdown(wait=True)

    def _run(self, job_id: str) -> None:
        job = self._repo.get_job(job_id)
        if job is None:
            return
        try:
            job.status = "running"
            self._repo.update_job(job)

            for image_id in job.image_ids:
                try:
                    img = self._repo.get_image(image_id)
                    if img is None:
                        raise ValueError("圖片已被刪除")
                    if job.prompt:
                        self._pipeline.segment_text(img, job.prompt)
                    else:
                        self._pipeline.segment_image(img)
                except Exception as exc:  # 單張失敗記錄下來，不讓整批中斷
                    logger.warning("批量分割失敗 job=%s image=%s: %s", job_id, image_id, exc)
                    job.failed.append({"image_id": image_id, "error": str(exc)})
                job.done += 1
                self._repo.update_job(job)

            job.status = "done"
            self._repo.update_job(job)
        except Exception:
            # 迴圈外的非預期錯誤（如存檔失敗）：job 不能停在 running，
            # 否則前端會永遠輪詢、409 防重複也會永遠擋住新 job
            logger.exception("批量分割非預期失敗 job=%s", job_id)
            try:
                job.status = "interrupted"
                self._repo.update_job(job)
            except Exception:
                logger.exception("寫回 interrupted 狀態也失敗 job=%s", job_id)
