from flask import Blueprint, request, abort
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from app.config import config

bp = Blueprint("linebot", __name__)

handler = WebhookHandler(config.line_channel_secret)



@bp.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")

    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"