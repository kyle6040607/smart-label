"""Email 驗證碼寄送。

未設定 SMTP_HOST 時走開發模式：驗證碼直接印在伺服器 log，方便本機測試。
正式環境用 SMTP（Gmail 應用程式密碼 / SendGrid SMTP 皆可），走 STARTTLS。
"""
from __future__ import annotations

import smtplib
from email.message import EmailMessage

from app.config import Config


def send_verification_code(cfg: Config, to_email: str, code: str) -> None:
    """寄出 6 位數驗證碼。寄送失敗會往外拋例外，由呼叫端決定怎麼回應使用者。"""
    subject = "Smart Label 驗證碼"
    body = (
        f"你的 Smart Label 驗證碼是：{code}\n\n"
        f"驗證碼 {cfg.otp_ttl_seconds // 60} 分鐘內有效。\n"
        "如果這不是你本人的操作，請忽略這封信。"
    )

    # 開發模式：沒設定 SMTP 就印在 log
    if not cfg.smtp_host:
        print(f"[DEV] 驗證碼寄給 {to_email}: {code}")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.mail_from or cfg.smtp_user
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15) as smtp:
        smtp.starttls()
        if cfg.smtp_user:
            smtp.login(cfg.smtp_user, cfg.smtp_password)
        smtp.send_message(msg)
