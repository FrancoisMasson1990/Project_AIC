from typing import Any

from app_context import AppContext
from flask import render_template

from aic.dashboard.dashapp import create_dash_app

flask_app, model, online = AppContext.app()
dash_app = create_dash_app(flask_app, "/dash/", model, online)


@flask_app.route("/")
def home_index() -> Any:
    return render_template("index.html")


@flask_app.route("/dash")
def dash_index():
    return dash_app.index()


if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port="8000", debug=True)
