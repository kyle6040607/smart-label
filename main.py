"""啟動進入點。
    uv run python main.py        # 開發伺服器
    # 或部署時用 gunicorn: gunicorn "app:create_app()"
"""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        debug=os.getenv("DEBUG", "1") == "1",
    )
