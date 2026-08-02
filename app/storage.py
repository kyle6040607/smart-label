"""本機與 Google Cloud Storage 共用的檔案儲存介面。"""
from __future__ import annotations

import shutil
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    BinaryIO,
    ContextManager,
    Iterator,
    Protocol,
)
from urllib.parse import quote

if TYPE_CHECKING:
    from app.config import Config


class StorageBackend(Protocol):
    def save_bytes(
        self,
        object_name: str,
        data: bytes,
        content_type: str | None = None,
    ) -> str: ...

    def save_file(
        self,
        object_name: str,
        source: Path,
        content_type: str | None = None,
    ) -> str: ...

    def read_bytes(self, reference: str) -> bytes: ...

    def open_reader(self, reference: str) -> BinaryIO: ...

    def open_writer(
        self,
        object_name: str,
        content_type: str | None = None,
    ) -> ContextManager[tuple[BinaryIO, str]]: ...

    def delete(self, reference: str) -> bool: ...

    def delete_prefix(self, object_prefix: str) -> int: ...

    def exists(self, reference: str) -> bool: ...


def _safe_object_name(object_name: str) -> PurePosixPath:
    """正規化物件名稱並阻擋絕對路徑與目錄穿越。"""
    normalized = object_name.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"無效的儲存路徑：{object_name!r}")
    return path


class LocalStorage:
    """把物件分類映射到既有的 data/uploads、masks、tasks 目錄。"""

    def __init__(
        self,
        data_dir: Path,
        upload_dir: Path,
        mask_dir: Path,
        dataset_dir: Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.category_roots = {
            "images": Path(upload_dir),
            "masks": Path(mask_dir),
            "datasets": (
                Path(dataset_dir)
                if dataset_dir is not None
                else Path(data_dir) / "tasks"
            ),
        }

    def _target(self, object_name: str) -> Path:
        key = _safe_object_name(object_name)
        category = key.parts[0]
        relative_parts = key.parts[1:]

        if category in self.category_roots:
            root = self.category_roots[category]
            if not relative_parts:
                raise ValueError(f"儲存路徑缺少檔名：{object_name!r}")
            return root.joinpath(*relative_parts)

        return self.data_dir.joinpath(*key.parts)

    def save_bytes(
        self,
        object_name: str,
        data: bytes,
        content_type: str | None = None,
    ) -> str:
        del content_type
        target = self._target(object_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
        return str(target)

    def read_bytes(self, reference: str) -> bytes:
        return Path(reference).read_bytes()

    def save_file(
        self,
        object_name: str,
        source: Path,
        content_type: str | None = None,
    ) -> str:
        del content_type
        source = Path(source)
        target = self._target(object_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        return str(target)

    def open_reader(self, reference: str) -> BinaryIO:
        return Path(reference).open("rb")

    @contextmanager
    def open_writer(
        self,
        object_name: str,
        content_type: str | None = None,
    ) -> Iterator[tuple[BinaryIO, str]]:
        del content_type
        target = self._target(object_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")

        try:
            with temporary.open("wb") as writer:
                yield writer, str(target)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def delete(self, reference: str) -> bool:
        path = Path(reference)
        existed = path.is_file()
        path.unlink(missing_ok=True)
        return existed

    def delete_prefix(self, object_prefix: str) -> int:
        """刪除受控儲存根目錄下的 attempt 前綴。"""
        normalized = str(_safe_object_name(object_prefix))
        target = self._target(normalized)
        resolved_target = target.resolve()
        allowed_roots = {
            self.data_dir.resolve(),
            *(root.resolve() for root in self.category_roots.values()),
        }
        if not any(
            resolved_target != root
            and resolved_target.is_relative_to(root)
            for root in allowed_roots
        ):
            raise ValueError(f"拒絕刪除儲存根目錄：{object_prefix!r}")
        if target.is_file() or target.is_symlink():
            target.unlink(missing_ok=True)
            return 1
        if not target.is_dir():
            return 0
        deleted = sum(1 for path in target.rglob("*") if path.is_file())
        try:
            shutil.rmtree(target)
        except FileNotFoundError:
            return 0
        return deleted

    def exists(self, reference: str) -> bool:
        return Path(reference).is_file()


class GCSStorage:
    """Google Cloud Storage 後端，reference 使用 gs://bucket/object。"""

    def __init__(
        self,
        bucket_name: str,
        project_id: str = "",
        client=None,
    ):
        if not bucket_name:
            raise ValueError("USE_GCS=1 時必須設定 GCS_BUCKET_NAME 或 GCS_BUCKET")

        if client is None:
            from google.cloud import storage as google_storage

            client = google_storage.Client(project=project_id or None)

        self.client = client
        self.bucket_name = bucket_name
        self.bucket = client.bucket(bucket_name)

    def _object_name(self, reference: str) -> str:
        prefix = f"gs://{self.bucket_name}/"
        if not reference.startswith(prefix):
            raise ValueError(
                f"GCS 路徑不屬於目前 Bucket {self.bucket_name!r}：{reference}"
            )
        return str(_safe_object_name(reference[len(prefix):]))

    def save_bytes(
        self,
        object_name: str,
        data: bytes,
        content_type: str | None = None,
    ) -> str:
        key = str(_safe_object_name(object_name))
        self.bucket.blob(key).upload_from_string(
            data,
            content_type=content_type,
        )
        return f"gs://{self.bucket_name}/{key}"

    def read_bytes(self, reference: str) -> bytes:
        from google.api_core.exceptions import NotFound

        try:
            return self.bucket.blob(
                self._object_name(reference)
            ).download_as_bytes()
        except NotFound as exc:
            raise FileNotFoundError(reference) from exc

    def save_file(
        self,
        object_name: str,
        source: Path,
        content_type: str | None = None,
    ) -> str:
        key = str(_safe_object_name(object_name))
        self.bucket.blob(key).upload_from_filename(
            str(source),
            content_type=content_type,
        )
        return f"gs://{self.bucket_name}/{key}"

    def open_reader(self, reference: str) -> BinaryIO:
        return self.bucket.blob(
            self._object_name(reference)
        ).open("rb")

    @contextmanager
    def open_writer(
        self,
        object_name: str,
        content_type: str | None = None,
    ) -> Iterator[tuple[BinaryIO, str]]:
        key = str(_safe_object_name(object_name))
        blob = self.bucket.blob(key)
        blob.content_type = content_type
        reference = f"gs://{self.bucket_name}/{key}"

        with blob.open(
            "wb",
            chunk_size=8 * 1024 * 1024,
            ignore_flush=True,
        ) as writer:
            yield writer, reference

    def generate_download_url(
        self,
        reference: str,
        download_name: str,
        expires_minutes: int = 10,
    ) -> str:
        from google.auth.credentials import Signing
        from google.auth.transport.requests import Request

        blob = self.bucket.blob(self._object_name(reference))
        credentials = getattr(self.client, "_credentials", None)
        signing_options = {}

        if credentials is not None:
            if isinstance(credentials, Signing):
                signing_options["credentials"] = credentials
            else:
                credentials.refresh(Request())
                service_account_email = getattr(
                    credentials,
                    "service_account_email",
                    "",
                )
                if not service_account_email:
                    raise AttributeError(
                        "目前憑證沒有可供 IAM signBlob 使用的 "
                        "service_account_email"
                    )
                signing_options.update(
                    service_account_email=service_account_email,
                    access_token=credentials.token,
                )

        disposition = (
            "attachment; "
            f"filename*=UTF-8''{quote(download_name)}"
        )
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expires_minutes),
            method="GET",
            response_type="application/zip",
            response_disposition=disposition,
            **signing_options,
        )

    def delete(self, reference: str) -> bool:
        from google.api_core.exceptions import NotFound

        try:
            self.bucket.blob(
                self._object_name(reference)
            ).delete()
        except NotFound:
            return False
        return True

    def delete_prefix(self, object_prefix: str) -> int:
        """刪除目前 Bucket 中指定物件前綴，不接受 gs:// reference。"""
        from google.api_core.exceptions import NotFound

        prefix = f"{str(_safe_object_name(object_prefix)).rstrip('/')}/"
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        deleted = 0
        for blob in blobs:
            try:
                blob.delete()
            except NotFound:
                continue
            deleted += 1
        return deleted

    def exists(self, reference: str) -> bool:
        return self.bucket.blob(
            self._object_name(reference)
        ).exists()


class StorageService:
    """依設定決定新檔存放處，並依 reference 讀取新舊檔案。"""

    def __init__(
        self,
        local: LocalStorage,
        gcs: GCSStorage | None = None,
        use_gcs: bool = False,
    ):
        if use_gcs and gcs is None:
            raise ValueError("已啟用 GCS，但尚未設定 GCS 儲存後端")
        self.local = local
        self.gcs = gcs
        self.use_gcs = use_gcs

    @property
    def backend_name(self) -> str:
        return "gcs" if self.use_gcs else "local"

    def save_bytes(
        self,
        object_name: str,
        data: bytes,
        content_type: str | None = None,
    ) -> str:
        backend = self.gcs if self.use_gcs else self.local
        assert backend is not None
        return backend.save_bytes(object_name, data, content_type)

    def _backend_for(self, reference: str) -> StorageBackend:
        if reference.startswith("gs://"):
            if self.gcs is None:
                raise RuntimeError(
                    "資料指向 GCS，但目前未設定 GCS_BUCKET_NAME 或 GCS_BUCKET"
                )
            return self.gcs
        return self.local

    def read_bytes(self, reference: str) -> bytes:
        return self._backend_for(reference).read_bytes(reference)

    def save_file(
        self,
        object_name: str,
        source: Path,
        content_type: str | None = None,
    ) -> str:
        backend = self.gcs if self.use_gcs else self.local
        assert backend is not None
        return backend.save_file(object_name, source, content_type)

    def open_reader(self, reference: str) -> BinaryIO:
        return self._backend_for(reference).open_reader(reference)

    @contextmanager
    def open_writer(
        self,
        object_name: str,
        content_type: str | None = None,
    ) -> Iterator[tuple[BinaryIO, str]]:
        backend = self.gcs if self.use_gcs else self.local
        assert backend is not None
        with backend.open_writer(
            object_name,
            content_type,
        ) as result:
            yield result

    def generate_download_url(
        self,
        reference: str,
        download_name: str,
        expires_minutes: int = 10,
    ) -> str | None:
        if not reference.startswith("gs://"):
            return None
        if self.gcs is None:
            raise RuntimeError(
                "資料指向 GCS，但目前未設定 GCS 儲存後端"
            )
        return self.gcs.generate_download_url(
            reference,
            download_name,
            expires_minutes,
        )

    def delete(self, reference: str) -> bool:
        return self._backend_for(reference).delete(reference)

    def delete_prefix(self, object_prefix: str) -> int:
        backend = self.gcs if self.use_gcs else self.local
        assert backend is not None
        return backend.delete_prefix(object_prefix)

    def exists(self, reference: str) -> bool:
        return self._backend_for(reference).exists(reference)


def build_storage(config: Config) -> StorageService:
    local = LocalStorage(
        data_dir=config.data_dir,
        upload_dir=config.upload_dir,
        mask_dir=config.mask_dir,
    )

    gcs = None
    if config.gcs_bucket_name:
        gcs = GCSStorage(
            bucket_name=config.gcs_bucket_name,
            project_id=config.gcs_project_id,
        )

    return StorageService(
        local=local,
        gcs=gcs,
        use_gcs=config.use_gcs,
    )


def require_configured_storage(
    config: Config,
    storage: StorageService | None,
) -> StorageService:
    """確保 USE_GCS 與實際寫入 backend 一致，禁止靜默寫回本機。"""
    if storage is None:
        if config.use_gcs:
            raise RuntimeError(
                "USE_GCS=1，但沒有提供 StorageService"
            )
        return StorageService(
            local=LocalStorage(
                data_dir=config.data_dir,
                upload_dir=config.upload_dir,
                mask_dir=config.mask_dir,
            )
        )

    if config.use_gcs and storage.backend_name != "gcs":
        raise RuntimeError(
            "USE_GCS=1，但 StorageService backend 不是 GCS"
        )

    return storage
