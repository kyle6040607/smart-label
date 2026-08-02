from app.config import Config
from app.models import AnnotationTask
from app.repository import Repository
from app.routes import line_bot
from app.services import task_notifier


def test_push_task_download_builds_button_message(monkeypatch):
    pushed_requests = []

    class FakeApiClient:
        def __init__(self, configuration):
            self.configuration = configuration

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class FakeMessagingApi:
        def __init__(self, api_client):
            self.api_client = api_client

        def push_message(self, request):
            pushed_requests.append(request)

    monkeypatch.setattr(line_bot, "ApiClient", FakeApiClient)
    monkeypatch.setattr(line_bot, "MessagingApi", FakeMessagingApi)

    assert line_bot.push_task_download(
        line_user_id="U-button-test",
        task_id="task-1",
        dataset_version=2,
        download_url="https://example.com/download.zip",
    )

    request = pushed_requests[0]
    message = request.messages[0]

    assert request.to == "U-button-test"
    assert message.alt_text == "標註任務已完成，可以下載 ZIP"
    assert message.template.actions[0].uri == "https://example.com/download.zip"


def test_notify_task_completed_updates_notified_version(
    tmp_path,
    monkeypatch,
):
    repo = Repository(tmp_path / "store.json")
    config = Config(public_base_url="https://example.com")
    task = AnnotationTask(
        line_user_id="U-notification-test",
        prompt="cat",
        image_ids=["image-1"],
        processed_image_ids=["image-1"],
        dataset_version=1,
        status="completed",
        dataset_zip_path=str(tmp_path / "dataset_v1.zip"),
        download_token="download-token",
    )
    repo.add_task(task)

    sent_messages = []

    def fake_push_task_download(**kwargs):
        sent_messages.append(kwargs)
        return True

    monkeypatch.setattr(
        task_notifier, "push_task_download", fake_push_task_download
    )

    assert task_notifier.notify_task_completed(repo, config, task)
    assert task.notified_dataset_version == 1
    assert sent_messages == [{
        "line_user_id": "U-notification-test",
        "task_id": task.id,
        "dataset_version": 1,
        "download_url": (
            "https://example.com/liff/tasks/"
            f"{task.id}/download?token=download-token"
            "&openExternalBrowser=1"
        ),
    }]
    assert not task_notifier.notify_task_completed(repo, config, task)
    assert len(sent_messages) == 1


def test_notify_final_failure_only_once(tmp_path, monkeypatch):
    repo = Repository(tmp_path / "store.json")
    task = repo.add_task(
        AnnotationTask(
            line_user_id="U-final-failure",
            status="failed",
            attempt_count=3,
            last_error="model crashed",
        )
    )
    calls = []
    monkeypatch.setattr(
        task_notifier,
        "push_task_failed",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    assert task_notifier.notify_task_failed(repo, Config(), task)
    assert not task_notifier.notify_task_failed(repo, Config(), task)
    assert len(calls) == 1
    assert calls[0]["line_user_id"] == "U-final-failure"


def test_failed_notification_does_not_mark_version(
    tmp_path,
    monkeypatch,
):
    repo = Repository(tmp_path / "store.json")
    config = Config(public_base_url="https://example.com")
    task = AnnotationTask(
        line_user_id="U-notification-failure",
        prompt="dog",
        image_ids=["image-1"],
        processed_image_ids=["image-1"],
        dataset_version=2,
        notified_dataset_version=1,
        status="completed",
        dataset_zip_path=str(tmp_path / "dataset_v2.zip"),
    )
    repo.add_task(task)

    monkeypatch.setattr(
        task_notifier, "push_task_download", lambda **kwargs: False
    )

    assert not task_notifier.notify_task_completed(repo, config, task)
    assert task.notified_dataset_version == 1
