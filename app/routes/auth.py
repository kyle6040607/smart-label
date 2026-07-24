"""登入 / 登出。

以 Flask session 記住登入狀態，密碼比對走 werkzeug 雜湊（永不比對明文）。
使用者資料存在 Repository（目前 JSON 檔，之後可抽換成 MySQL / MongoDB）。
"""
from __future__ import annotations

import hashlib
import re
import secrets
import time
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
from werkzeug.security import check_password_hash, generate_password_hash

from app.models import User
from app.routes import get_config, get_repo
from app.services import line_login, mailer

bp = Blueprint("auth", __name__)


def get_authenticated_user() -> User | None:
    """取得目前登入使用者；帳號失效時只移除登入欄位。"""
    user_id = session.get("user_id")
    if not user_id:
        return None

    user = get_repo().get_user(user_id)
    if user is None:
        # LINE OAuth 的 state / nonce 仍可能在同一個 session 中，不能整包清除。
        session.pop("user_id", None)
        session.pop("username", None)
    return user


def login_required(view):
    """保護需要登入才能看的頁面 / API，未登入導向登入頁。"""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if get_authenticated_user() is None:
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


@bp.get("/login")
def login():
    if get_authenticated_user() is not None:
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

    # 有 Email 但還沒完成驗證碼驗證 → 補寄驗證碼並導去驗證頁
    # （沒有 Email 的舊帳號 / 預設帳號不受影響）
    if user.email and not user.email_verified:
        err = _issue_otp(user)
        if err:
            return render_template("login.html", error=err), 502
        get_repo().update_user(user)
        session["pending_verify_user_id"] = user.id
        return redirect(url_for("auth.verify"))

    session.clear()
    session["user_id"] = user.id
    session["username"] = user.username

    return redirect(url_for("index"))


@bp.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))


# ---------- 註冊 ----------
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _email_domain_exists(email: str) -> bool:
    """檢查 Email 網域是否收信（DNS MX 查詢，查不到 MX 再退而查 A 紀錄）。

    能當場擋掉打錯的網域（如 gmial.com 打成 asdfasdf123.com）。
    DNS 查詢本身失敗（逾時等）時放行，不因網路問題擋掉合法使用者；
    網域存在但信箱帳號不存在的情況，交給驗證碼把關。
    """
    import dns.resolver

    domain = email.rsplit("@", 1)[-1]
    resolver = dns.resolver.Resolver()
    resolver.lifetime = 3.0
    for record_type in ("MX", "A"):
        try:
            resolver.resolve(domain, record_type)
            return True
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
            continue
        except Exception as e:  # noqa: BLE001 DNS 服務異常時放行
            print(f"Email 網域 DNS 查詢失敗（放行）: {e}")
            return True
    return False


def _validate_password(pw: str) -> str | None:
    """檢查密碼規範，通過回傳 None，否則回傳錯誤訊息。

    規範：至少 8 個字元、一個大寫字母、一個小寫字母、一個特殊字元。
    """
    if len(pw) < 8:
        return "密碼長度至少 8 個字元"
    if not re.search(r"[A-Z]", pw):
        return "密碼必須包含至少一個大寫字母"
    if not re.search(r"[a-z]", pw):
        return "密碼必須包含至少一個小寫字母"
    if not re.search(r"[^A-Za-z0-9]", pw):
        return "密碼必須包含至少一個特殊字元（如 !@#$%）"
    return None


@bp.get("/register")
def register():
    if get_authenticated_user() is not None:
        return redirect(url_for("index"))
    return render_template("register.html", error=None, form={})


@bp.post("/register")
def do_register():
    repo = get_repo()
    ac = (request.form.get("ac") or "").strip()
    pw = request.form.get("pw") or ""
    mail = (request.form.get("mail") or "").strip()

    # 保留使用者已填的欄位（密碼除外），驗證失敗時不用重打
    form = {"ac": ac, "mail": mail}

    def fail(message: str, status: int = 400):
        return render_template("register.html", error=message, form=form), status

    if not ac:
        return fail("請輸入帳號")
    if not mail or not _EMAIL_RE.match(mail):
        return fail("Email 格式不正確")
    if not _email_domain_exists(mail):
        return fail("Email 網域不存在，請確認是否拼錯")

    pw_error = _validate_password(pw)
    if pw_error:
        return fail(pw_error)

    # 帳號重複：已驗證的帳號（或 LINE / 無 Email 的舊帳號）才擋；
    # 未完成 Email 驗證的殘留帳號（例如打錯信箱）允許重新註冊覆蓋
    existing = repo.get_user_by_username(ac)
    replaceable = (
        existing is not None
        and bool(existing.password_hash)
        and bool(existing.email)
        and not existing.email_verified
    )
    if existing is not None and not replaceable:
        return fail("這個帳號已被使用", 409)

    # Email 重複：只擋已通過驗證的帳號（未驗證的殘留帳號永遠無法啟用，不佔用 Email）
    if any(
        u.email and u.email.lower() == mail.lower() and u.email_verified
        for u in repo.list_users()
    ):
        return fail("這個 Email 已被註冊", 409)

    if replaceable:
        user = existing
        user.password_hash = generate_password_hash(pw)  # 只存雜湊，不落地明文
        user.email = mail
    else:
        user = User(
            username=ac,
            password_hash=generate_password_hash(pw),
            email=mail,
            role="user",
        )

    # 先確認驗證信寄得出去才寫入資料庫；寄不出去就不留下帳號
    err = _issue_otp(user)
    if err:
        return fail(err, 502)

    if replaceable:
        repo.update_user(user)
    else:
        repo.add_user(user)

    # 通過 Email 驗證才算註冊完成並登入
    session["pending_verify_user_id"] = user.id
    return redirect(url_for("auth.verify"))


# ---------- Email 驗證碼 ----------
def _otp_hash(user_id: str, code: str) -> str:
    """驗證碼不存明碼；以 user_id 當鹽做 sha256。"""
    return hashlib.sha256(f"{user_id}:{code}".encode()).hexdigest()


def _issue_otp(user: User) -> str | None:
    """產生 6 位數驗證碼設定到 user 上並寄信（不寫入資料庫）。

    成功回傳 None；寄送失敗回傳錯誤訊息，此時呼叫端不應把 user 寫入資料庫，
    避免留下永遠無法驗證的殘留帳號。
    """
    cfg = get_config()
    code = f"{secrets.randbelow(10**6):06d}"
    user.otp_hash = _otp_hash(user.id, code)
    user.otp_expires = time.time() + cfg.otp_ttl_seconds
    user.otp_attempts = 0
    try:
        mailer.send_verification_code(cfg, user.email, code)
    except Exception as e:  # noqa: BLE001 寄信細節不外洩給使用者
        print(f"驗證信寄送失敗: {e}")
        return "驗證信寄送失敗，請確認 Email 是否正確，或稍後再試"
    return None


def _mask_email(email: str) -> str:
    """遮罩 Email 顯示：ab****@gmail.com。"""
    local, _, domain = email.partition("@")
    visible = local[:2] if len(local) > 2 else local[:1]
    return f"{visible}****@{domain}"


def _get_pending_user() -> User | None:
    user_id = session.get("pending_verify_user_id")
    if not user_id:
        return None
    return get_repo().get_user(user_id)


@bp.get("/verify")
def verify():
    user = _get_pending_user()
    if user is None:
        return redirect(url_for("auth.login"))
    return render_template(
        "verify.html", error=None, masked_email=_mask_email(user.email)
    )


@bp.post("/verify")
def do_verify():
    cfg = get_config()
    repo = get_repo()
    user = _get_pending_user()
    if user is None:
        return redirect(url_for("auth.login"))

    def fail(message: str, status: int = 400):
        return render_template(
            "verify.html", error=message, masked_email=_mask_email(user.email)
        ), status

    code = (request.form.get("code") or "").strip()

    if not user.otp_hash or time.time() > user.otp_expires:
        return fail("驗證碼已過期，請按「重寄驗證碼」")
    if user.otp_attempts >= cfg.otp_max_attempts:
        return fail("嘗試次數過多，請按「重寄驗證碼」")

    if not code or _otp_hash(user.id, code) != user.otp_hash:
        user.otp_attempts += 1
        repo.update_user(user)
        remaining = cfg.otp_max_attempts - user.otp_attempts
        if remaining <= 0:
            return fail("嘗試次數過多，請按「重寄驗證碼」")
        return fail(f"驗證碼錯誤，還可再試 {remaining} 次")

    # 驗證通過：標記完成、清掉驗證碼、正式登入
    user.email_verified = True
    user.otp_hash = ""
    user.otp_expires = 0.0
    user.otp_attempts = 0
    repo.update_user(user)

    session.clear()
    session["user_id"] = user.id
    session["username"] = user.username
    return redirect(url_for("index"))


@bp.post("/verify/resend")
def resend_code():
    user = _get_pending_user()
    if user is None:
        return redirect(url_for("auth.login"))
    err = _issue_otp(user)
    if err:
        return render_template(
            "verify.html", error=err, masked_email=_mask_email(user.email)
        ), 502
    get_repo().update_user(user)
    return render_template(
        "verify.html",
        error=None,
        masked_email=_mask_email(user.email),
        resent=True,
    )


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
