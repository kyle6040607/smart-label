"""YOLOv26x-seg 實例分割訓練服務。

支援多租戶隔離 (user_id / owner_id, project_id)，從 Cloud SQL / Repository 讀取標註，
於容器暫存目錄 (/tmp) 動態生成 YOLO Segmentation 的標註檔 (.txt) 與 data.yaml，
並以預抓好的 models/yolo26x-seg.pt 權重檔執行訓練。

訓練產出之模型檔名自動帶有【日期時間】格式（如 yolo26x_seg_20260801_141710.pt），
訓練結束後透過 try-finally 區塊 100% 強制清除所有 /tmp 暫存檔。
"""
from __future__ import annotations

import datetime
import io
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.models import AnnotationTask, ImageRecord, Segment
    from app.repository import Repository
    from app.storage import StorageService

logger = logging.getLogger(__name__)


def _read_bytes(reference: str, storage: StorageService | None) -> bytes:
    if storage is not None:
        return storage.read_bytes(reference)
    return Path(reference).read_bytes()


def _read_image(
    reference: str,
    flags: int,
    storage: StorageService | None,
) -> np.ndarray | None:
    try:
        data = _read_bytes(reference, storage)
    except (FileNotFoundError, OSError):
        return None
    return cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        flags,
    )


def _image_size(
    image: ImageRecord,
    segs: list[Segment],
    storage: StorageService | None,
) -> tuple[int, int]:
    """回傳 (w, h)：優先用紀錄值，沒有就從遮罩推回。"""
    if image.width and image.height:
        return image.width, image.height
    if segs and segs[0].mask_path:
        m = _read_image(
            segs[0].mask_path,
            cv2.IMREAD_GRAYSCALE,
            storage,
        )
        if m is not None:
            h, w = m.shape[:2]
            return w, h
    return 640, 640


def _mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    """二值遮罩 → 多邊形輪廓，每個輪廓是攤平的 [x1,y1,x2,y2,...]。"""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys: list[list[float]] = []
    for c in contours:
        if cv2.contourArea(c) < 1:
            continue
        poly = c.reshape(-1, 2).astype(float)
        if len(poly) >= 3:
            polys.append(poly.flatten().tolist())
    return polys


def prepare_yolo_dataset(
    repo: Repository,
    storage: StorageService | None,
    user_id: str,
    project_id: str | None,
    work_dir: Path,
) -> tuple[Path, list[str]]:
    """依據 user_id 與 project_id 從 Repository 撈出已標註片段，並建立 YOLO segmentation 資料集。

    回傳 (yaml_path, labels_list)。
    """
    images_dir = work_dir / "images"
    labels_dir = work_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    labeled_segs = repo.list_labeled_segments_by_project(user_id, project_id)
    if not labeled_segs:
        raise ValueError(f"使用者 {user_id!r} 專案 {project_id!r} 沒有已完成標註的片段")

    labels = sorted({s.final_label for s in labeled_segs if s.final_label})
    cls_idx = {name: i for i, name in enumerate(labels)}

    by_image: dict[str, list[Segment]] = {}
    for s in labeled_segs:
        by_image.setdefault(s.image_id, []).append(s)

    for image_id, segs in by_image.items():
        image = repo.get_image(image_id)
        if not image:
            continue

        w, h = _image_size(image, segs, storage)
        safe_stem = f"{image.id}_{Path(image.filename).stem or 'img'}"
        suffix = Path(image.filename).suffix or ".jpg"
        img_dest = images_dir / f"{safe_stem}{suffix}"

        # 複製/解出圖片檔至暫存區
        try:
            img_bytes = _read_bytes(image.path, storage)
            img_dest.write_bytes(img_bytes)
        except Exception as e:
            logger.warning("無法讀取圖片 %s: %s", image.path, e)
            continue

        lines: list[str] = []
        for s in segs:
            if not s.final_label or s.final_label not in cls_idx:
                continue
            mask = _read_image(s.mask_path, cv2.IMREAD_GRAYSCALE, storage)
            if mask is None:
                continue

            for poly in _mask_to_polygons(mask):
                coords = " ".join(
                    f"{(v / w if i % 2 == 0 else v / h):.6f}"
                    for i, v in enumerate(poly)
                )
                lines.append(f"{cls_idx[s.final_label]} {coords}")

        label_dest = labels_dir / f"{safe_stem}.txt"
        body = "\n".join(lines)
        label_dest.write_text(body + ("\n" if body else ""), encoding="utf-8")

    yaml_content = (
        f"path: {work_dir.resolve().as_posix()}\n"
        "train: images\n"
        "val: images\n"
        f"nc: {len(labels)}\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(labels))
    )

    yaml_path = work_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    return yaml_path, labels


def resolve_device(device_setting: str = "auto") -> str:
    """自動偵測可用硬體：優先使用 GPU (cuda:0)，若無法使用則自動 fallback 退回 CPU。"""
    try:
        import torch
        if device_setting in ("0", "cuda", "auto", "gpu"):
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("偵測到可用 GPU (%s)，啟用 GPU (cuda:0) 執行訓練", gpu_name)
                return "0"
            else:
                logger.info("目前環境未支援/未安裝 CUDA GPU，自動回退 (fallback) 至 CPU 執行訓練")
                return "cpu"
    except Exception as e:
        logger.warning("檢測 GPU 支援時發生例外 (%s)，自動回退至 CPU 執行", e)
    return device_setting if device_setting != "auto" else "cpu"


def train_yolov26x_seg(
    repo: Repository,
    storage: StorageService | None,
    task: AnnotationTask,
    *,
    epochs: int = 5,
    imgsz: int = 640,
    device: str = "auto",
    base_model_path: str = "models/yolo26x-seg.pt",
) -> AnnotationTask:
    """執行 YOLOv26x-seg 訓練。

    訓練完畢後，產出之模型檔名改為日期時間格式 yolo26x_seg_YYYYMMDD_HHMMSS.pt，
    寫回 Cloud SQL / DB 的 best_model_path，並在 finally 區塊強制清理所有 /tmp 暫存檔。
    """
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    model_filename = f"yolo26x_seg_{timestamp_str}.pt"
    target_device = resolve_device(device)

    # 建立臨時工作目錄
    temp_dir = tempfile.mkdtemp(prefix=f"yolo_train_{task.id}_")
    work_dir = Path(temp_dir)

    try:
        task.status = "processing"
        repo.update_task(task)

        yaml_path, labels = prepare_yolo_dataset(
            repo,
            storage,
            user_id=task.user_id,
            project_id=task.project_id,
            work_dir=work_dir,
        )

        from ultralytics import YOLO, settings

        models_dir = (Path(__file__).resolve().parent.parent.parent / "models").resolve()
        models_dir.mkdir(exist_ok=True)
        settings.update({"weights_dir": str(models_dir)})

        # 確定基礎預訓練權重存在，優先使用 models/ 資料夾內的權重檔，避免下載至主資料夾
        weights_file = Path(base_model_path)
        if not weights_file.is_file():
            models_dir_file = models_dir / weights_file.name
            if models_dir_file.is_file():
                weights_file = models_dir_file
            else:
                # 依序尋找 models/ 內的預設備用權重
                for fallback_name in ["yolo26n.pt", "yolo26x-seg.pt", "yolov8x-worldv2.pt"]:
                    fb_path = models_dir / fallback_name
                    if fb_path.is_file():
                        weights_file = fb_path
                        break
                else:
                    # 若都不存在，指定將檔名放置於 models/ 目錄下，觸發 Ultralytics 自動下載至 models/
                    weights_file = models_dir / weights_file.name

        model = YOLO(str(weights_file))

        train_runs_dir = work_dir / "runs"
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            device=target_device,
            amp=(target_device != "cpu"),
            project=str(train_runs_dir),
            name="exp",
            exist_ok=True,
            verbose=False,
            workers=0,
        )

        # 找出訓練出來的 best.pt
        save_dir = getattr(results, "save_dir", None)
        best_pt = Path(save_dir) / "weights" / "best.pt" if save_dir else train_runs_dir / "exp" / "weights" / "best.pt"

        if not best_pt.is_file():
            # 若沒跑出 best.pt，抓任何有 .pt 的權重檔
            pt_files = list(work_dir.glob("**/*.pt"))
            if pt_files:
                best_pt = pt_files[0]
            else:
                raise FileNotFoundError(f"訓練完成，但找不到權重檔 best.pt：{train_runs_dir}")

        # 上傳/儲存模型權重檔至 Storage
        user_id = task.user_id or "default_user"
        proj_id = task.project_id or "default_project"
        object_name = f"users/{user_id}/projects/{proj_id}/models/{model_filename}"

        if storage is not None:
            best_model_ref = storage.save_file(object_name, best_pt)
        else:
            local_dest = Path("models") / model_filename
            local_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(best_pt, local_dest)
            best_model_ref = str(local_dest)

        task.best_model_path = best_model_ref
        task.status = "completed"
        task.error_message = ""
        task.updated_at = time.time()
        repo.update_task(task)
        logger.info("YOLOv26x-seg 訓練成功！模型儲存於 %s", best_model_ref)
        return task

    except Exception as exc:
        logger.error("YOLOv26x-seg 訓練失敗：task_id=%s, error=%s", task.id, exc, exc_info=True)
        task.status = "failed"
        task.error_message = str(exc)
        task.updated_at = time.time()
        repo.update_task(task)
        raise
    finally:
        # 強制完全刪除 /tmp 中的所有暫存訓練資料
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("已清理暫存訓練目錄: %s", work_dir)
