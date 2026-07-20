"""登入 / 登出。

以 Flask session 記住登入狀態，密碼比對走 werkzeug 雜湊（永不比對明文）。
使用者資料存在 Repository（目前 JSON 檔，之後可抽換成 MySQL / MongoDB）。
"""
from __future__ import annotations

import secrets
from functools import wraps

from flask import (
    Blueprint,
    abort,
    current_app,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash

from app.models import User
from app.routes import get_config, get_repo
from app.services import line_login

bp = Blueprint("auth", __name__)


def login_required(view):
    """保護需要登入才能看的頁面 / API，未登入導向登入頁。"""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def api_login_required(view):
    """API 版登入保護：未登入直接回 401，不導向登入頁（fetch 跟隨 302 會拿到 HTML）。"""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            abort(401, "請先登入")
        return view(*args, **kwargs)

    return wrapped


@bp.get("/login")
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))
    return render_template("login.html", error=None)


@bp.post("/login")
def do_login():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    user = get_repo().get_user_by_username(username)
    # password_hash 為空代表 LINE-only 帳號，不接受密碼登入
    if user is None or not user.password_hash or not check_password_hash(
        user.password_hash, password
    ):
        return render_template("login.html", error="帳號或密碼錯誤"), 401

    session.clear()
    session["user_id"] = user.id
    session["username"] = user.username

    return redirect(url_for("index"))


@bp.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))


# ---------- LINE Login ----------
def _redirect_uri() -> str:
    """LINE 導回的 callback 網址，必須與 LINE Login channel 設定的完全相符。"""
    cfg = get_config()
    return cfg.line_login_redirect_uri or url_for("auth.line_callback", _external=True)


@bp.get("/login/line")
def line_start():
    """把使用者導去 LINE 授權。若已登入，這趟流程視為「綁定 LINE 到現有帳號」。"""
    cfg = get_config()
    if not cfg.line_login_channel_id or not cfg.line_login_channel_secret:
        return render_template("login.html", error="尚未設定 LINE Login（請填環境變數）"), 500

    state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(24)
    session["line_oauth_state"] = state
    session["line_oauth_nonce"] = nonce
    session["line_oauth_bind"] = bool(session.get("user_id"))

    url = line_login.authorize_url(cfg.line_login_channel_id, _redirect_uri(), state, nonce)
    return redirect(url)


@bp.get("/login/line/callback")
def line_callback():
    cfg = get_config()
    repo = get_repo()

    # 1) 擋 CSRF：state 必須與導出前存進 session 的相符
    saved_state = session.pop("line_oauth_state", None)
    nonce = session.pop("line_oauth_nonce", None)
    bind = session.pop("line_oauth_bind", False)
    if not request.args.get("state") or request.args.get("state") != saved_state:
        return render_template("login.html", error="LINE 登入驗證失敗（state 不符）"), 400
    if request.args.get("error"):
        return render_template("login.html", error="已取消 LINE 登入"), 400
    code = request.args.get("code")
    if not code:
        return render_template("login.html", error="LINE 未回傳授權碼"), 400

    # 2) 授權碼換 token，再請 LINE 驗證 id_token 並解出使用者資料
    try:
        token = line_login.exchange_token(
            code, _redirect_uri(), cfg.line_login_channel_id, cfg.line_login_channel_secret
        )
        profile = line_login.verify_id_token(token["id_token"], cfg.line_login_channel_id, nonce)
    except Exception:  # noqa: BLE001 對外呼叫失敗一律回同一則錯誤，不外洩細節
        return render_template("login.html", error="無法與 LINE 完成登入，請稍後再試"), 502

    line_user_id = profile.get("sub")
    if not line_user_id:
        return render_template("login.html", error="LINE 未提供使用者識別"), 502
    display_name = profile.get("name", "")
    avatar_url = profile.get("picture", "")

    # 3-a) 綁定模式：把 LINE 綁到目前登入的帳號
    if bind and session.get("user_id"):
        current = repo.get_user(session["user_id"])
        owner = repo.get_user_by_line_id(line_user_id)
        if owner is not None and (current is None or owner.id != current.id):
            return render_template("login.html", error="這個 LINE 帳號已綁定其他帳號"), 409
        if current is not None:
            current.line_user_id = line_user_id
            current.display_name = current.display_name or display_name
            current.avatar_url = avatar_url or current.avatar_url
            repo.update_user(current)
            return redirect(url_for("index"))

    # 3-b) 登入模式：找已綁定的帳號，沒有就自動建立一個 LINE-only 帳號
    user = repo.get_user_by_line_id(line_user_id)
    if user is None:
        user = repo.add_user(
            User(
                username=f"line_{line_user_id[:10]}",  # 佔位帳號名，之後可讓使用者改
                line_user_id=line_user_id,
                display_name=display_name,
                avatar_url=avatar_url,
                role="user",
            )
        )

    session.clear()
    session["user_id"] = user.id
    session["username"] = user.display_name or user.username
    return redirect(url_for("index"))
