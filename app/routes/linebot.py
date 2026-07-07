from flask import Blueprint, request
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError

bp = Blueprint("linebot", __name__)


@bp.route("/callback", methods=["POST"])
def callback():
    print("====== 收到 LINE Webhook ======")
    print(request.get_json())

    return "OK"