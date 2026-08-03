"""LINE Messaging API Webhook 與主動推播。

聊天室不接收圖片或文字指令。使用者透過 Rich Menu 開啟 LIFF，
任務完成後由背景 Worker 主動推播下載通知。
"""

from flask import Blueprint, abort, request
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    ButtonsTemplate,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    ReplyMessageRequest,
    TemplateMessage,
    TextMessage,
    URIAction,
)
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import FollowEvent

from app.config import config


bp = Blueprint("linebot", __name__)
handler = WebhookHandler(config.line_channel_secret)
configuration = Configuration(access_token=config.line_channel_access_token)


def reply_text(reply_token: str, text: str) -> bool:
    """回覆單一文字訊息。"""
    if not reply_token:
        return False

    try:
        with ApiClient(configuration) as api_client:
            MessagingApi(api_client).reply_message(
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[TextMessage(text=text)],
                )
            )
        return True
    except Exception as exc:
        print(f"LINE 回覆失敗：{exc}")
        return False


def push_task_download(
    line_user_id: str,
    task_id: str,
    dataset_version: int,
    download_url: str,
) -> bool:
    """推播任務完成通知與 ZIP 下載按鈕。"""
    if not line_user_id or not task_id or dataset_version <= 0 or not download_url:
        return False

    try:
        message = TemplateMessage(
            altText="標註任務已完成，可以下載 ZIP",
            template=ButtonsTemplate(
                title="標註任務已完成",
                text=f"任務 {task_id} 的 YOLO 資料集 v{dataset_version} 已準備完成。",
                actions=[URIAction(label="下載 ZIP", uri=download_url)],
            ),
        )

        with ApiClient(configuration) as api_client:
            MessagingApi(api_client).push_message(
                PushMessageRequest(to=line_user_id, messages=[message])
            )
        return True
    except Exception as exc:
        print(f"LINE 任務完成通知失敗：{exc}")
        return False


def push_task_no_export(
    line_user_id: str,
    task_id: str,
    excluded_count: int,
) -> bool:
    """推播已完成但沒有可匯出 ZIP 的結果。"""
    if not line_user_id or not task_id:
        return False
    detail = (
        f"共有 {excluded_count} 個結果低於匯出信心門檻。"
        if excluded_count > 0
        else "沒有偵測到可匯出的結果。"
    )
    try:
        with ApiClient(configuration) as api_client:
            MessagingApi(api_client).push_message(
                PushMessageRequest(
                    to=line_user_id,
                    messages=[
                        TextMessage(
                            text=(
                                f"標註任務 {task_id} 已完成，但未產生 ZIP。\n"
                                f"{detail}\n"
                                "請從 Rich Menu 開啟任務紀錄查看未通過縮圖。"
                            )
                        )
                    ],
                )
            )
        return True
    except Exception as exc:
        print(f"LINE 無匯出結果通知發送失敗：{exc}")
        return False


def push_task_failed(
    line_user_id: str,
    task_id: str,
    error_message: str,
) -> bool:
    """推播達最大嘗試次數後的最終失敗通知。"""
    if not line_user_id or not task_id:
        return False
    message = error_message.strip() or "處理圖片時發生錯誤"
    if len(message) > 300:
        message = f"{message[:297]}..."
    try:
        with ApiClient(configuration) as api_client:
            MessagingApi(api_client).push_message(
                PushMessageRequest(
                    to=line_user_id,
                    messages=[
                        TextMessage(
                            text=(
                                f"標註任務 {task_id} 處理失敗。\n"
                                f"系統已完成自動重試，請稍後重新建立任務。\n"
                                f"原因：{message}"
                            )
                        )
                    ],
                )
            )
        return True
    except Exception as exc:
        print(f"LINE 任務失敗通知發送失敗：{exc}")
        return False

@bp.post("/callback")
def callback():
    """驗證 LINE 簽章並轉交 Webhook 事件。"""
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


@handler.add(FollowEvent)
def handle_follow(event):
    """使用者加入好友時，引導他改由 Rich Menu 操作。"""
    reply_text(
        event.reply_token,
        "歡迎使用 Seer 👋\n\n"
        "請使用聊天室下方的 Rich Menu：\n"
        "・建立標註任務\n"
        "・查看標註任務\n\n"
        "圖片與標註內容都會在 LIFF 頁面中設定。",
    )
