from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Blueprint, request, abort
from PIL import Image

from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    ImageMessageContent,
    TextMessageContent,
    FollowEvent,
)
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
)

from app.config import config
from app.models import ImageRecord
from app.routes import get_config, get_repo, get_pipeline


bp = Blueprint("linebot", __name__)

handler = WebhookHandler(config.line_channel_secret)

configuration = Configuration(
    access_token=config.line_channel_access_token
)

# 限制背景處理的併發數，避免大量訊息湧入時執行緒數量失控
executor = ThreadPoolExecutor(max_workers=4)


# ============================================================
# LINE 訊息
# ============================================================


def reply_text(reply_token: str, text: str) -> None:
    """用 reply_token 立即回覆（限時效內、單次使用）。"""
    try:
        with ApiClient(configuration) as api_client:
            MessagingApi(api_client).reply_message(
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[
                        TextMessage(text=text),
                    ],
                )
            )

    except Exception as e:
        print(f"reply_message 失敗: {e}")


def push_text(user_id: str, text: str) -> None:
    """主動推播訊息給 LINE 使用者。"""
    try:
        with ApiClient(configuration) as api_client:
            MessagingApi(api_client).push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[
                        TextMessage(text=text),
                    ],
                )
            )

    except Exception as e:
        print(f"push_message 失敗: {e}")


# ============================================================
# 圖片清理
# ============================================================


def _cleanup_image(repo, image_id: str | None) -> None:
    """刪除 Repository 圖片紀錄與磁碟實體檔案。"""
    if not image_id:
        return

    image_record = repo.get_image(image_id)

    if image_record is None:
        return

    for path in repo.delete_image(image_id):
        Path(path).unlink(missing_ok=True)


def _cleanup_images(
    repo,
    image_ids: list[str],
) -> None:
    """批次清除圖片。"""
    for image_id in image_ids:
        _cleanup_image(
            repo,
            image_id,
        )


# ============================================================
# LINE Webhook
# ============================================================


@bp.route("/callback", methods=["POST"])
def callback():
    """接收 LINE Webhook 並驗證簽章。"""
    signature = request.headers.get(
        "X-Line-Signature"
    )

    if not signature:
        abort(400)

    body = request.get_data(as_text=True)

    try:
        handler.handle(
            body,
            signature,
        )
        
    except InvalidSignatureError:
        abort(400)

    return "OK"


@handler.add(FollowEvent)
def handle_follow(event):
    """使用者加入 LINE 官方帳號時顯示產品使用說明。"""

    print("====== 使用者加入 LINE 官方帳號 ======")

    reply_text(
        event.reply_token,
        (
            "歡迎使用 Smart Label 智慧影像標註助手 👋\n\n"
            "我可以幫你批次接收圖片，"
            "並依照你的文字需求進行 AI 影像搜尋與標註 🤖\n\n"
            "使用方式很簡單：\n\n"
            "1️⃣ 傳送一張或多張圖片 📷\n"
            "2️⃣ 圖片全部傳完後，輸入「完成」\n"
            "3️⃣ 確認圖片數量後，告訴我想搜尋的物件\n\n"
            "例如：\n"
            "「貓咪」\n"
            "「安全帽」\n"
            "「汽車」\n\n"
            "現在就可以直接傳圖片給我 🚀"
        ),
    )

# ============================================================
# LINE Session
# ============================================================


def try_process_session(
    user_id: str,
    reply_token: str,
    repo,
    pipeline,
) -> None:
    """
    檢查 LINE Session。

    圖片與 Prompt 都完成時，
    才啟動背景處理。
    """

    status, session = (
        repo.try_consume_line_session(user_id)
    )

    # --------------------------------------------------------
    # 沒有圖片
    # --------------------------------------------------------

    if status == "empty":
        reply_text(
            reply_token,
            "請先傳送至少一張圖片給我 📷",
        )
        return

    # --------------------------------------------------------
    # 持續收圖片：安靜不回覆
    # --------------------------------------------------------

    if status == "collecting_images":
        return

    # --------------------------------------------------------
    # 圖片已傳完，等待使用者確認
    # --------------------------------------------------------

    if status == "awaiting_confirmation":
        assert session is not None

        reply_text(
            reply_token,
            (
                f"你總共傳了 {len(session.image_ids)} 張圖片，確定了嗎？\n"
                "請輸入「確認」或「取消」。"
            ),
        )

        return

    # --------------------------------------------------------
    # 已確認，等待 Prompt
    # --------------------------------------------------------

    if status == "waiting_prompt":
        reply_text(
            reply_token,
            "已確認 ✅\n"
            "請告訴我想搜尋或標註什麼物件 ✏️",
        )

        return

    # --------------------------------------------------------
    # Ready
    # --------------------------------------------------------

    assert session is not None
    assert session.prompt is not None
    assert session.image_ids

    reply_text(
        reply_token,
        (
            f"已收到 {len(session.image_ids)} 張圖片和說明 ✅\n"
            "處理中，請稍候 ⏳"
        ),
    )

    executor.submit(
        process_and_notify,
        user_id,
        session.prompt,
        session.image_ids.copy(),
        repo,
        pipeline,
    )


# ============================================================
# 背景圖片處理
# ============================================================


def process_and_notify(
    user_id: str,
    prompt: str,
    image_ids: list[str],
    repo,
    pipeline,
) -> None:
    """背景處理多張圖片。"""

    print(
        "====== 開始處理任務（背景執行緒）======"
    )
    print(f"user_id: {user_id}")
    print(f"prompt: {prompt}")
    print(f"image_ids: {image_ids}")

    try:
        image_records = []

        for image_id in image_ids:
            image_record = repo.get_image(
                image_id
            )

            if image_record is None:
                raise ValueError(
                    f"找不到圖片 image_id={image_id}"
                )

            image_records.append(
                image_record
            )

        # TODO:
        # 之後接自然語言分割
        #
        # for image_record in image_records:
        #     segments = pipeline.segment_text(
        #         image_record,
        #         prompt,
        #     )

        image_info = "\n".join(
            (
                f"{index}. "
                f"{image_record.id} "
                f"({image_record.width} × "
                f"{image_record.height})"
            )
            for index, image_record
            in enumerate(
                image_records,
                start=1,
            )
        )

        result_text = (
            "圖片取得成功 ✅\n"
            f"圖片數量: {len(image_records)}\n"
            f"搜尋物件: {prompt}\n\n"
            "圖片資訊:\n"
            f"{image_info}"
        )

    except Exception as e:
        print(f"處理任務失敗: {e}")

        push_text(
            user_id,
            "處理過程發生錯誤，請稍後再試 🙏",
        )

        return

    push_text(
        user_id,
        result_text,
    )


# ============================================================
# 圖片訊息
# ============================================================


@handler.add(
    MessageEvent,
    message=ImageMessageContent,
)
def handle_image_message(event):
    """接收並儲存 LINE 圖片。"""

    print("====== 收到 LINE 圖片 ======")

    user_id = getattr(
        event.source,
        "user_id",
        None,
    )

    if not user_id:
        print("找不到 user_id")
        return

    cfg = get_config()
    repo = get_repo()
    pipeline = get_pipeline()

    message_id = event.message.id

    print(f"message_id: {message_id}")

    # --------------------------------------------------------
    # 從 LINE 下載圖片
    # --------------------------------------------------------

    try:
        with ApiClient(configuration) as api_client:
            image_bytes = (
                MessagingApiBlob(api_client)
                .get_message_content(
                    message_id=message_id,
                )
            )

    except Exception as e:
        print(
            f"取得圖片訊息失敗: {e}"
        )

        reply_text(
            event.reply_token,
            "抱歉，圖片下載失敗，請再試一次 🙏",
        )

        return

    # --------------------------------------------------------
    # 建立 ImageRecord
    # --------------------------------------------------------

    rec = ImageRecord(
        filename=f"line_{message_id}.jpg"
    )

    dest = (
        cfg.upload_dir
        / f"{rec.id}_{rec.filename}"
    )

    # --------------------------------------------------------
    # 寫入檔案
    # --------------------------------------------------------

    try:
        dest.write_bytes(image_bytes)

        with Image.open(dest) as image:
            rec.width, rec.height = (
                image.size
            )

    except Exception as e:
        print(f"圖片儲存失敗: {e}")

        dest.unlink(
            missing_ok=True,
        )

        reply_text(
            event.reply_token,
            "圖片儲存失敗，請再試一次 🙏",
        )

        return

    rec.path = str(dest)

    # --------------------------------------------------------
    # Repository 儲存圖片
    # --------------------------------------------------------

    repo.add_image(rec)

    print(
        f"圖片已存檔: "
        f"{rec.path} "
        f"（{rec.width}x{rec.height}）"
    )

    # --------------------------------------------------------
    # 加入 LINE Session
    # --------------------------------------------------------

    session, expired_image_ids, reopened = (
        repo.add_line_session_image(
            user_id,
            rec.id,
        )
    )

    # --------------------------------------------------------
    # 清除逾時 Session 的舊圖片
    # --------------------------------------------------------

    _cleanup_images(
        repo,
        expired_image_ids,
    )

    # --------------------------------------------------------
    # 回覆使用者：一般收圖階段安靜不回覆，
    # 只有「已經傳完了/確認過又傳新圖」這種狀態被打斷的情況才提醒
    # --------------------------------------------------------

    if reopened:
        reply_text(
            event.reply_token,
            (
                "收到新的圖片 📷\n"
                "已重新開啟圖片上傳。\n"
                f"目前共有 {len(session.image_ids)} 張圖片。\n"
                "傳完後請再次輸入「完成」。"
            ),
        )


# ============================================================
# 文字訊息
# ============================================================


@handler.add(
    MessageEvent,
    message=TextMessageContent,
)
def handle_text_message(event):
    """處理 LINE 文字訊息。"""

    print(
        "====== 收到 LINE 文字訊息 ======"
    )

    user_id = getattr(
        event.source,
        "user_id",
        None,
    )

    if not user_id:
        print("找不到 user_id")
        return

    text = event.message.text.strip()

    if not text:
        reply_text(
            event.reply_token,
            "文字內容不能是空的喔",
        )

        return

    print(f"文字訊息: {text}")

    repo = get_repo()
    pipeline = get_pipeline()
    # --------------------------------------------------------
    # Rich Menu：開始標註
    # --------------------------------------------------------

    if text == "開始標註":
        reply_text(
            event.reply_token,
            (
                "請開始傳送圖片 📷\n\n"
                "你可以一次傳送多張圖片。\n"
                "全部傳完後請輸入「完成」。"
            ),
        )
        return


    # --------------------------------------------------------
    # Rich Menu：使用教學
    # --------------------------------------------------------

    if text == "使用教學":
        reply_text(
            event.reply_token,
            (
                "📖 Smart Label 使用教學\n\n"
                "1️⃣ 傳送一張或多張圖片\n"
                "2️⃣ 圖片全部傳完後輸入「完成」\n"
                "3️⃣ 確認圖片數量\n"
                "4️⃣ 輸入想搜尋或標註的物件\n"
                "5️⃣ AI 自動處理圖片 🤖\n\n"
                "例如：\n"
                "貓咪\n"
                "安全帽\n"
                "汽車"
            ),
        )
        return


    # --------------------------------------------------------
    # Rich Menu：重新開始
    # --------------------------------------------------------

    if text == "重新開始":
        session = repo.get_line_session(user_id)

        image_ids = (
            session.image_ids.copy()
            if session is not None
            else []
        )

        repo.clear_line_session(user_id)

        _cleanup_images(
            repo,
            image_ids,
        )

        reply_text(
            event.reply_token,
            (
                "已重新開始 🔄\n\n"
                "舊的圖片已清除。\n"
                "請重新傳送圖片 📷"
            ),
        )

        return
    # --------------------------------------------------------
    # 使用者表示圖片已傳完
    # --------------------------------------------------------

    if text == "完成":
        session = (
            repo.mark_line_session_images_done(
                user_id
            )
        )

        if session is None:
            reply_text(
                event.reply_token,
                "目前還沒有收到圖片喔，請先傳圖片給我 📷",
            )

            return

        try_process_session(
            user_id,
            event.reply_token,
            repo,
            pipeline,
        )

        return

    # --------------------------------------------------------
    # 使用者確認圖片沒問題
    # --------------------------------------------------------

    if text == "確認":
        session = (
            repo.confirm_line_session_images(
                user_id
            )
        )

        if session is None:
            reply_text(
                event.reply_token,
                "目前沒有待確認的圖片，請先傳圖片並輸入「傳完了」",
            )

            return

        try_process_session(
            user_id,
            event.reply_token,
            repo,
            pipeline,
        )

        return

    # --------------------------------------------------------
    # 使用者取消，清空已傳的圖片
    # --------------------------------------------------------

    if text == "取消":
        removed_image_ids = (
            repo.reset_line_session_images(
                user_id
            )
        )

        _cleanup_images(
            repo,
            removed_image_ids,
        )

        reply_text(
            event.reply_token,
            "已取消，請重新傳送圖片 📷",
        )

        return

    # --------------------------------------------------------
    # 檢查 Session
    # --------------------------------------------------------

    session = repo.get_line_session(
        user_id
    )

    if session is None or not session.image_ids:
        reply_text(
            event.reply_token,
            (
                "請先傳送圖片給我 📷\n"
                "全部傳完後輸入「完成」。"
            ),
        )

        return

    # --------------------------------------------------------
    # 圖片仍在收集中：安靜不回覆
    # --------------------------------------------------------

    if not session.images_done:
        return

    # --------------------------------------------------------
    # 已傳完但還沒確認
    # --------------------------------------------------------

    if not session.confirmed:
        reply_text(
            event.reply_token,
            "請輸入「確認」或「取消」。",
        )

        return

    # --------------------------------------------------------
    # 儲存 Prompt
    # --------------------------------------------------------

    session = repo.set_line_session_prompt(
        user_id,
        text,
    )

    if session is None:
        reply_text(
            event.reply_token,
            "目前無法設定文字說明，請重新操作 🙏",
        )

        return

    # --------------------------------------------------------
    # 嘗試開始處理
    # --------------------------------------------------------

    try_process_session(
        user_id,
        event.reply_token,
        repo,
        pipeline,
    )
