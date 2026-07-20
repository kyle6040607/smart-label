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

from app.models import ImageRecord, Segment, LabelExample, User, LineSession, SegmentJob


class Repository:
    # LINE session 多久沒湊齊視為過期（秒）；過期視為新一輪，交由呼叫端回收舊圖
    SESSION_TTL_SECONDS = 600

    def __init__(self, db_file: Path):
        self.db_file = db_file
        self._lock = threading.Lock()
        self.images: dict[str, ImageRecord] = {}
        self.segments: dict[str, Segment] = {}
        self.examples: dict[str, LabelExample] = {}
        self.users: dict[str, User] = {}
        self.line_sessions: dict[str, LineSession] = {}
        self.jobs: dict[str, SegmentJob] = {}
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

    # ---------- LINE session（多圖累加 + 明確「傳完了」信號才收 prompt）----------
    def _fresh_session_or_expired(self, line_user_id: str) -> tuple[LineSession | None, list[str]]:
        """回傳（未過期的 session 或 None, 若過期要清掉的舊圖 id 列表）。"""
        s = self.line_sessions.get(line_user_id)
        if s is None:
            return None, []
        if time.time() - s.updated_at > self.SESSION_TTL_SECONDS:
            return None, list(s.image_ids)
        return s, []

    def get_line_session(self, line_user_id: str) -> LineSession | None:
        return self.line_sessions.get(line_user_id)

    def add_line_session_image(self, line_user_id: str, image_id: str) -> tuple[LineSession, list[str], bool]:
        """新增一張圖到 session。

        回傳 (session, 因逾時被清掉的舊圖 id 列表, reopened)。
        reopened=True 代表使用者先前已輸入「完成」（不管是否已確認），
        這次傳圖重新打開收圖狀態。
        """
        with self._lock:
            s, expired_ids = self._fresh_session_or_expired(line_user_id)
            if s is None:
                s = LineSession(line_user_id=line_user_id)
            reopened = s.images_done
            s.image_ids.append(image_id)
            s.images_done = False
            s.confirmed = False
            s.updated_at = time.time()
            self.line_sessions[line_user_id] = s
            self._save()
            return s, expired_ids, reopened

    def mark_line_session_images_done(self, line_user_id: str) -> LineSession | None:
        """使用者輸入「完成」，進入待確認狀態。若一張圖都沒傳，回傳 None。"""
        with self._lock:
            s = self.line_sessions.get(line_user_id)
            if s is None or not s.image_ids:
                return None
            s.images_done = True
            s.confirmed = False
            s.updated_at = time.time()
            self._save()
            return s

    def confirm_line_session_images(self, line_user_id: str) -> LineSession | None:
        """使用者輸入「確認」。若目前不是「待確認」狀態，回傳 None。"""
        with self._lock:
            s = self.line_sessions.get(line_user_id)
            if s is None or not s.images_done or s.confirmed:
                return None
            s.confirmed = True
            s.updated_at = time.time()
            self._save()
            return s

    def reset_line_session_images(self, line_user_id: str) -> list[str]:
        """使用者輸入「取消」，清空已傳的圖片，回傳要清掉的 image_id 列表。"""
        with self._lock:
            s = self.line_sessions.get(line_user_id)
            if s is None:
                return []
            old_ids = list(s.image_ids)
            s.image_ids = []
            s.images_done = False
            s.confirmed = False
            s.prompt = None
            s.updated_at = time.time()
            self._save()
            return old_ids

    def set_line_session_prompt(self, line_user_id: str, prompt: str) -> LineSession | None:
        """設定 prompt。若圖片還沒確認，回傳 None（呼叫端應先擋掉這個狀態）。"""
        with self._lock:
            s = self.line_sessions.get(line_user_id)
            if s is None or not s.confirmed:
                return None
            s.prompt = prompt
            s.updated_at = time.time()
            self._save()
            return s

    def try_consume_line_session(self, line_user_id: str) -> tuple[str, LineSession | None]:
        """檢查 session 狀態；圖文都到齊時原子性地取出並清除，避免重複觸發。

        回傳 status: "empty" | "collecting_images" | "awaiting_confirmation"
        | "waiting_prompt" | "ready"。
        """
        with self._lock:
            s = self.line_sessions.get(line_user_id)
            if s is None or not s.image_ids:
                return "empty", None
            if not s.images_done:
                return "collecting_images", s
            if not s.confirmed:
                return "awaiting_confirmation", s
            if s.prompt is None:
                return "waiting_prompt", s
            self.line_sessions.pop(line_user_id, None)
            self._save()
            return "ready", s

    def clear_line_session(self, line_user_id: str) -> None:
        with self._lock:
            self.line_sessions.pop(line_user_id, None)
            self._save()

    # ---------- 批量分割 job ----------
    def add_job(self, job: SegmentJob) -> SegmentJob:
        with self._lock:
            self.jobs[job.id] = job
            self._save()
        return job

    def get_job(self, job_id: str) -> SegmentJob | None:
        return self.jobs.get(job_id)

    def update_job(self, job: SegmentJob) -> SegmentJob:
        with self._lock:
            self.jobs[job.id] = job
            self._save()
        return job

    def list_jobs(self) -> list[SegmentJob]:
        """最近建立的在前，前端重整頁面後靠這個找回進行中的批量工作。"""
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    # ---------- 統計（給準確率曲線 / 省下工時用）----------
    def stats(self) -> dict:
        segs = list(self.segments.values())
        total = len(segs)
        need_review = sum(1 for s in segs if s.needs_review)
        reviewed = sum(1 for s in segs if s.reviewed)
        auto_accepted = total - need_review
        return {
            "total_segments": total,
            "auto_accepted": auto_accepted,
            "need_review": need_review,
            "reviewed": reviewed,
            # 自動接受比例 ≈ 省下的人工工時（提案第 3 頁）
            "auto_ratio": round(auto_accepted / total, 3) if total else 0.0,
            "num_examples": len(self.examples),
            "num_labels": len(self.labels()),
        }

    # ---------- 持久化 ----------
    def _save(self) -> None:
        payload = {
            "images": [i.to_dict() for i in self.images.values()],
            "segments": [s.to_dict() for s in self.segments.values()],
            "examples": [e.to_dict() for e in self.examples.values()],
            "users": [u.to_dict() for u in self.users.values()],
            "line_sessions": [s.to_dict() for s in self.line_sessions.values()],
            "jobs": [j.to_dict() for j in self.jobs.values()],
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
        for d in data.get("line_sessions", []):
            self.line_sessions[d["line_user_id"]] = LineSession(
                **{k: v for k, v in d.items() if k in LineSession.__dataclass_fields__}
            )
        for d in data.get("jobs", []):
            d.pop("total", None)  # 衍生欄位不還原
            job = SegmentJob(**{k: v for k, v in d.items() if k in SegmentJob.__dataclass_fields__})
            # 重啟後背景 thread 已不存在，殘留的未完成 job 標成 interrupted，
            # 否則前端會永遠輪詢一個不會前進的進度條
            if job.status in ("queued", "running"):
                job.status = "interrupted"
            self.jobs[job.id] = job
