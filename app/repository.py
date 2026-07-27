"""資料存取層（Repository）。

目前用「記憶體 + JSON 檔」當作最小可用儲存，介面刻意做成可抽換：
之後接 MySQL / MongoDB（提案第 9 頁）只要實作同一組方法即可，
上層 API 與 pipeline 完全不用改。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from app.models import AnnotationTask,ImageRecord, Segment, LabelExample, User


class Repository:
    def __init__(self, db_file: Path):
        self.db_file = db_file
        self._lock = threading.Lock()
        self.images: dict[str, ImageRecord] = {}
        self.segments: dict[str, Segment] = {}
        self.examples: dict[str, LabelExample] = {}
        self.users: dict[str, User] = {}
        self.tasks: dict[str, AnnotationTask] = {}
        self.parameters: dict[str, float] = {}
        self._load() 

    # ---------- 影像 ----------
    def add_image(self, img: ImageRecord) -> ImageRecord:
        with self._lock:
            self.images[img.id] = img
            self._save()
        return img

    def get_image(self, image_id: str) -> ImageRecord | None:
        return self.images.get(image_id)

    def list_images(self) -> list[ImageRecord]:
        return sorted(self.images.values(), key=lambda i: i.created_at)

    def delete_image(self, image_id: str) -> list[str]:
        """刪除一張圖，連帶清掉它所有的遮罩片段紀錄。

        回傳需要從磁碟刪除的檔案路徑（原圖 + 各遮罩 PNG），
        實際刪檔交給上層處理（Repository 只管資料，不碰檔案系統）。
        由片段衍生的種子範例（examples）保留——那是已學到的知識，
        刪圖不該讓模型失憶。
        """
        with self._lock:
            img = self.images.pop(image_id, None)
            if img is None:
                return []
            paths = [img.path]
            for seg_id in [s.id for s in self.segments.values() if s.image_id == image_id]:
                seg = self.segments.pop(seg_id)
                if seg.mask_path:
                    paths.append(seg.mask_path)
            self._save()
        return [p for p in paths if p]

    def delete_images_batch(self, image_ids: list[str]) -> list[str]:
        """批次刪除多張圖與對應的遮罩片段，並在最後一次性存檔。"""
        paths = []
        with self._lock:
            for image_id in image_ids:
                img = self.images.pop(image_id, None)
                if img is None:
                    continue
                if img.path:
                    paths.append(img.path)
                seg_ids = [s.id for s in self.segments.values() if s.image_id == image_id]
                for seg_id in seg_ids:
                    seg = self.segments.pop(seg_id, None)
                    if seg and seg.mask_path:
                        paths.append(seg.mask_path)
            self._save()
        return [p for p in paths if p]

    # ---------- 標註任務 ----------
    # 建立任務
    def add_task(
        self,
        task: AnnotationTask,
    ) -> AnnotationTask:
        with self._lock:
            self.tasks[task.id] = task
            self._save()

        return task
    # 依ID 取回任務
    def get_task(
        self,
        task_id: str,
    ) -> AnnotationTask | None:
        return self.tasks.get(task_id)

    def list_tasks_by_line_user_id(
        self,
        line_user_id: str,
    ) -> list[AnnotationTask]:
        """依更新時間由新到舊取得 LINE 使用者的任務。"""
        return sorted(
            [
                task
                for task in self.tasks.values()
                if task.line_user_id == line_user_id
            ],
            key=lambda task: task.updated_at,
            reverse=True,
        )
    def claim_next_pending_task(
        self,
    ) -> AnnotationTask | None:
        with self._lock:
            pending_tasks = [
                task
                for task in self.tasks.values()
                if task.status == "pending"
            ]

            if not pending_tasks:
                return None

            task = min(
                pending_tasks,
                key=lambda item: item.created_at,
            )

            task.status = "processing"
            task.error_message = ""
            task.updated_at = time.time()

            self._save()

            return task
    # 背景執行
    def update_task(
        self,
        task: AnnotationTask,
    ) -> AnnotationTask:
        with self._lock:
            task.updated_at = time.time()
            self.tasks[task.id] = task
            self._save()

        return task
    # 讓 LIFF 任務在 Web 綁定 LINE 後自動歸戶
    def assign_tasks_to_user(
        self,
        line_user_id: str,
        user_id: str,
    ) -> int:
        assigned_count = 0

        with self._lock:
            for task in self.tasks.values():
                if task.line_user_id != line_user_id:
                    continue
                if task.user_id and task.user_id != user_id:
                    continue
                if not task.user_id:
                    task.user_id = user_id
                    task.updated_at = time.time()
                    assigned_count += 1
                if task.user_id == user_id:
                    for image_id in task.image_ids:
                        image = self.images.get(image_id)
                        if image is not None and not image.owner_id:
                            image.owner_id = user_id

            # 即使任務早已歸戶，也可能需要補上新版的圖片 owner_id。
            self._save()

        return assigned_count

    # ---------- 遮罩片段 ----------
    def add_segment(self, seg: Segment) -> Segment:
        with self._lock:
            self.segments[seg.id] = seg
            self._save()
        return seg

    def get_segment(self, seg_id: str) -> Segment | None:
        return self.segments.get(seg_id)

    def list_segments(self, image_id: str | None = None) -> list[Segment]:
        segs = self.segments.values()
        if image_id is not None:
            segs = [s for s in segs if s.image_id == image_id]
        return list(segs)

    def list_review_queue(self) -> list[Segment]:
        """待人工審核的低信心片段（提案第 8 頁標紅送審）。"""
        return [s for s in self.segments.values() if s.needs_review and not s.reviewed]

    def delete_segment(self, seg_id: str) -> str | None:
        """刪掉一個片段（切壞/不要的），回傳要刪的遮罩檔路徑。"""
        with self._lock:
            seg = self.segments.pop(seg_id, None)
            if seg is None:
                return None
            self._save()
        return seg.mask_path or None

    def delete_segments_batch(self, seg_ids: list[str]) -> list[str]:
        """批次刪除多個遮罩片段，並在最後一次性存檔。"""
        paths = []
        with self._lock:
            for seg_id in seg_ids:
                seg = self.segments.pop(seg_id, None)
                if seg and seg.mask_path:
                    paths.append(seg.mask_path)
            self._save()
        return [p for p in paths if p]

    def update_segment(self, seg: Segment) -> Segment:
        with self._lock:
            self.segments[seg.id] = seg
            self._save()
        return seg

    # ---------- few-shot 範例 ----------
    def add_example(self, ex: LabelExample) -> LabelExample:
        with self._lock:
            self.examples[ex.id] = ex
            self._save()
        return ex

    def list_examples(self) -> list[LabelExample]:
        return list(self.examples.values())

    def labels(self) -> list[str]:
        return sorted({ex.label for ex in self.examples.values()})

    def delete_label(self, label: str) -> int:
        """刪掉某類別的所有種子範例（標錯類別時用），回傳刪除的範例數。

        連帶把用這個錯誤類別人工標過的片段退回送審——類別都錯了，
        那些標記也不該留在匯出資料裡。回訓由上層 pipeline 負責。
        """
        with self._lock:
            ids = [eid for eid, ex in self.examples.items() if ex.label == label]
            for eid in ids:
                del self.examples[eid]
            for seg in self.segments.values():
                if seg.human_label == label:
                    seg.human_label = None
                    seg.reviewed = False
                    seg.needs_review = True
            self._save()
        return len(ids)

    # ---------- 使用者 / 登入 ----------
    def add_user(self, user: User) -> User:
        with self._lock:
            self.users[user.id] = user
            self._save()
        return user

    def get_user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        for u in self.users.values():
            if u.username == username:
                return u
        return None

    def get_user_by_line_id(self, line_user_id: str) -> User | None:
        for u in self.users.values():
            if u.line_user_id and u.line_user_id == line_user_id:
                return u
        return None

    def update_user(self, user: User) -> User:
        with self._lock:
            self.users[user.id] = user
            self._save()
        return user

    def list_users(self) -> list[User]:
        return list(self.users.values())

    # ---------- 統計（給準確率曲線 / 省下工時用）----------
    def stats(self) -> dict:
        segs = list(self.segments.values())
        total = len(segs)
        need_review = sum(1 for s in segs if s.needs_review)
        reviewed = sum(1 for s in segs if s.reviewed)
        auto_accepted = total - need_review

        label_counts = {}
        for s in segs:
            lbl = s.final_label
            if lbl:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

        return {
            "total_segments": total,
            "auto_accepted": auto_accepted,
            "need_review": need_review,
            "reviewed": reviewed,
            # 自動接受比例 ≈ 省下的人工工時（提案第 3 頁）
            "auto_ratio": round(auto_accepted / total, 3) if total else 0.0,
            "num_examples": len(self.examples),
            "num_labels": len(self.labels()),
            "label_counts": label_counts,
        }

    # ---------- 持久化 ----------
    def _save(self) -> None:
        payload = {
            "images": [i.to_dict() for i in self.images.values()],
            "segments": [s.to_dict() for s in self.segments.values()],
            "examples": [e.to_dict() for e in self.examples.values()],
            "users": [u.to_dict() for u in self.users.values()],
            "tasks": [task.to_dict() for task in self.tasks.values()],
            "parameters": self.parameters,
        }
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.db_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.db_file)

    def _load(self) -> None:
        if not self.db_file.exists():
            return
        data = json.loads(self.db_file.read_text(encoding="utf-8"))
        for d in data.get("images", []):
            self.images[d["id"]] = ImageRecord(**{k: v for k, v in d.items() if k in ImageRecord.__dataclass_fields__})
        for d in data.get("segments", []):
            d.pop("final_label", None)  # 衍生欄位不還原
            self.segments[d["id"]] = Segment(**{k: v for k, v in d.items() if k in Segment.__dataclass_fields__})
        for d in data.get("examples", []):
            self.examples[d["id"]] = LabelExample(**{k: v for k, v in d.items() if k in LabelExample.__dataclass_fields__})
        for d in data.get("users", []):
            self.users[d["id"]] = User(**{k: v for k, v in d.items() if k in User.__dataclass_fields__})
        for d in data.get("tasks", []):
            task_data = { key: value for key, value in d.items() if key in AnnotationTask.__dataclass_fields__}
            task = AnnotationTask(**task_data)
            if ("processed_image_ids" not in d and task.status == "completed"):
                task.processed_image_ids = list( task.image_ids )
            if ( "dataset_version" not in d and task.status == "completed" and task.dataset_zip_path):
                task.dataset_version = 1
            self.tasks[task.id] = task
        self.parameters = data.get("parameters", {})

    def get_parameters(self) -> dict[str, float]:
        with self._lock:
            return dict(self.parameters)

    def set_parameter(self, key: str, value: float) -> None:
        with self._lock:
            self.parameters[key] = value
            self._save()
