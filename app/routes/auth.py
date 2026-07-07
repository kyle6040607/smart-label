"""登入 / 登出。

以 Flask session 記住登入狀態，密碼比對走 werkzeug 雜湊（永不比對明文）。
使用者資料存在 Repository（目前 JSON 檔，之後可抽換成 MySQL / MongoDB）。
"""
from __future__ import annotations

from functools import wraps

from flask import (
    Blueprint,
    current_app,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash

from app.routes import get_repo

bp = Blueprint("auth", __name__)


def login_required(view):
    """保護需要登入才能看的頁面 / API，未登入導向登入頁。"""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("auth.login", next=request.path))
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
    if user is None or not check_password_hash(user.password_hash, password):
        return render_template("login.html", error="帳號或密碼錯誤"), 401

    session.clear()
    session["user_id"] = user.id
    session["username"] = user.username

    # 只允許站內相對路徑，避免 open redirect
    nxt = request.args.get("next") or request.form.get("next") or ""
    if not nxt.startswith("/"):
        nxt = url_for("index")
    return redirect(nxt)


@bp.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))
