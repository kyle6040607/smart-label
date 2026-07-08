"""LINE Login（OAuth 2.0 / OpenID Connect）最小客戶端。

只用標準庫 urllib，不引入額外套件。負責三件事：
1. authorize_url  — 組授權 URL，把使用者導去 LINE 同意畫面
2. exchange_token — 用 LINE 導回的授權碼換 access_token / id_token
3. verify_id_token — 交給 LINE 驗證 id_token 並解出使用者資料
                     （用官方 verify 端點，免自行驗 JWT 簽章）

文件：https://developers.line.biz/en/docs/line-login/integrate-line-login/
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

AUTHORIZE_URL = "https://access.line.me/oauth2/v2.1/authorize"
TOKEN_URL = "https://api.line.me/oauth2/v2.1/token"
VERIFY_URL = "https://api.line.me/oauth2/v2.1/verify"


def authorize_url(channel_id: str, redirect_uri: str, state: str, nonce: str) -> str:
    """組出要把使用者導去的 LINE 授權網址。

    state / nonce 由呼叫端隨機產生並存進 session，回呼時用來擋 CSRF / replay。
    scope 用 "profile openid"：拿得到 userId、顯示名稱、大頭貼（email 需另申請權限）。
    """
    q = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": channel_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "scope": "profile openid",
            "nonce": nonce,
        }
    )
    return f"{AUTHORIZE_URL}?{q}"


def _post_form(url: str, data: dict[str, str]) -> dict:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 (固定 https 端點)
        return json.loads(resp.read().decode())


def exchange_token(
    code: str, redirect_uri: str, channel_id: str, channel_secret: str
) -> dict:
    """用授權碼換 token，回傳含 access_token 與 id_token 的字典。"""
    return _post_form(
        TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": channel_id,
            "client_secret": channel_secret,
        },
    )


def verify_id_token(id_token: str, channel_id: str, nonce: str | None = None) -> dict:
    """交給 LINE 驗證 id_token，回傳解出的使用者資料。

    重要欄位：sub（LINE userId，帳號綁定的鍵）、name（顯示名稱）、picture（大頭貼）。
    帶入 nonce 讓 LINE 一併比對，擋掉重放攻擊。
    """
    data = {"id_token": id_token, "client_id": channel_id}
    if nonce:
        data["nonce"] = nonce
    return _post_form(VERIFY_URL, data)
