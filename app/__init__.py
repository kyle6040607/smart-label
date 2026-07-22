"""Flask 應用。

create_app() 建立 app、初始化 Repository 與 Pipeline，註冊 API blueprint。
單一進入點，方便測試與部署（Docker / GCP）。
"""
from __future__ import annotations

from flask import Flask, jsonify, render_template, request, session
from werkzeug.security import generate_password_hash

from app.config import Config, config as default_config
from app.models import User
from app.services.pipeline import Pipeline
from app.repository import Repository


def _seed_default_user(repo: Repository, cfg: Config) -> None:
    """首次啟動時建立預設帳號（sa/sa），密碼以雜湊值儲存。"""
    if repo.get_user_by_username(cfg.default_admin_user) is not None:
        return
    repo.add_user(
        User(
            username=cfg.default_admin_user,
            password_hash=generate_password_hash(cfg.default_admin_password),
            role="admin",
        )
    )


def create_app(config: Config | None = None) -> Flask:
    cfg = config or default_config
    cfg.ensure_dirs()

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_content_length
    app.config["SECRET_KEY"] = cfg.secret_key

    # 共用單例：repo 與 pipeline 掛在 app 上，blueprint 透過 current_app 取用
    app.smart_config = cfg          # type: ignore[attr-defined]
    app.repo = Repository(cfg.db_file)  # type: ignore[attr-defined]
    app.pipeline = Pipeline(cfg, app.repo)  # type: ignore[attr-defined]

    _seed_default_user(app.repo, cfg)

    from app.routes.auth import bp as auth_bp, get_authenticated_user, login_required
    from app.routes.images import bp as images_bp
    from app.routes.segment import bp as segment_bp
    from app.routes.labels import bp as labels_bp
    from app.routes.review import bp as review_bp
    from app.routes.export import bp as export_bp
    from app.routes.line_bot import bp as linebot_bp
    from app.routes.liff import bp as liff_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(images_bp)
    app.register_blueprint(segment_bp)
    app.register_blueprint(labels_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(linebot_bp)
    app.register_blueprint(liff_bp)

    @app.before_request
    def require_api_login():
        """所有 /api 路由都必須有對應到有效使用者的登入 session。"""
        if request.path != "/api" and not request.path.startswith("/api/"):
            return None

        if get_authenticated_user() is not None:
            return None

        return jsonify(
            {
                "error": "authentication_required",
                "message": "請先登入後再使用 API",
            }
        ), 401

    @app.get("/")
    @login_required
    def index():
        user = app.repo.get_user(session.get("user_id"))  # type: ignore[attr-defined]
        return render_template(
            "index.html",
            display_name=session.get("username", ""),
            line_bound=bool(user and user.line_user_id),
        )

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    return app
