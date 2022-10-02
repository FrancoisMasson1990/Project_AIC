from typing import Any

from app_context import AppContext
from flask import render_template

from aic.dashboard.dashapp import create_dash_app

server, model, online = AppContext.app()
app = create_dash_app(server, "/dash/", model, online)


@server.route("/")
def dash_app() -> Any:
    return render_template("index.html")


@server.route("/dash")
def index():
    return app.index()


if __name__ == "__main__":
    server.run(host="0.0.0.0", port="8000", debug=True)
