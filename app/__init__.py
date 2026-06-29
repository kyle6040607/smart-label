"""Flask 應用。

create_app() 建立 app、初始化 Repository 與 Pipeline，註冊 API blueprint。
單一進入點，方便測試與部署（Docker / GCP）。
"""
from __future__ import annotations

from flask import Flask, render_template

from app.config import Config, config as default_config
from app.services.pipeline import Pipeline
from app.repository import Repository


def create_app(config: Config | None = None) -> Flask:
    cfg = config or default_config
    cfg.ensure_dirs()

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_content_length

    # 共用單例：repo 與 pipeline 掛在 app 上，blueprint 透過 current_app 取用
    app.smart_config = cfg          # type: ignore[attr-defined]
    app.repo = Repository(cfg.db_file)  # type: ignore[attr-defined]
    app.pipeline = Pipeline(cfg, app.repo)  # type: ignore[attr-defined]

    from app.routes.images import bp as images_bp
    from app.routes.segment import bp as segment_bp
    from app.routes.labels import bp as labels_bp
    from app.routes.review import bp as review_bp
    from app.routes.export import bp as export_bp

    app.register_blueprint(images_bp)
    app.register_blueprint(segment_bp)
    app.register_blueprint(labels_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(export_bp)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    return app
