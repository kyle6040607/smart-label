"""資料存取層（Repository）。

目前用「記憶體 + JSON 檔」當作最小可用儲存，介面刻意做成可抽換：
之後接 MySQL / MongoDB（提案第 9 頁）只要實作同一組方法即可，
上層 API 與 pipeline 完全不用改。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from app.models import ImageRecord, Segment, LabelExample, User


class Repository:
    def __init__(self, db_file: Path):
        self.db_file = db_file
        self._lock = threading.Lock()
        self.images: dict[str, ImageRecord] = {}
        self.segments: dict[str, Segment] = {}
        self.examples: dict[str, LabelExample] = {}
        self.users: dict[str, User] = {}
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
