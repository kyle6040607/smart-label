from app.config import Config
from app.services import cloud_run_jobs


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.raise_called = False

    def raise_for_status(self):
        self.raise_called = True

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, credentials, response):
        self.credentials = credentials
        self.response = response
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_trigger_task_worker_is_disabled_without_job_name(tmp_path, monkeypatch):
    config = Config(base_dir=tmp_path)
    config.cloud_run_task_job_name = ""
    monkeypatch.setattr(
        cloud_run_jobs.google.auth,
        "default",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    assert cloud_run_jobs.trigger_task_worker(config) == ""


def test_trigger_task_worker_calls_cloud_run_jobs_api(tmp_path, monkeypatch):
    config = Config(base_dir=tmp_path)
    config.cloud_run_task_job_name = "smart-label-task-worker"
    config.cloud_run_task_job_project_id = "smart-label-501610"
    config.cloud_run_task_job_region = "asia-east1"
    config.cloud_run_task_job_trigger_timeout_seconds = 7
    credentials = object()
    response = FakeResponse(
        {"name": "projects/example/locations/asia-east1/operations/operation-1"}
    )
    session = FakeSession(credentials, response)
    requested_scopes = []

    def fake_default(*, scopes):
        requested_scopes.extend(scopes)
        return credentials, config.cloud_run_task_job_project_id

    monkeypatch.setattr(cloud_run_jobs.google.auth, "default", fake_default)
    monkeypatch.setattr(
        cloud_run_jobs,
        "AuthorizedSession",
        lambda supplied_credentials: (
            session
            if supplied_credentials is credentials
            else (_ for _ in ()).throw(AssertionError(supplied_credentials))
        ),
    )

    operation_name = cloud_run_jobs.trigger_task_worker(config)

    assert operation_name.endswith("/operations/operation-1")
    assert requested_scopes == [
        "https://www.googleapis.com/auth/cloud-platform"
    ]
    assert session.calls == [
        (
            "https://run.googleapis.com/v2/projects/smart-label-501610/"
            "locations/asia-east1/jobs/smart-label-task-worker:run",
            {"json": {}, "timeout": 7},
        )
    ]
    assert response.raise_called is True
