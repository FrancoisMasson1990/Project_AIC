from typing import Any

from app_context import AppContext
from flask import render_template

from aic.dashboard.dashapp_inference import create_dash_inference
from aic.dashboard.dashapp_label import create_dash_label

flask_app, model, online = AppContext.app()
dash_inference = create_dash_inference(
    flask_app, "/dash_inference/", model, online
)
dash_labeling = create_dash_label(flask_app, "/dash_labeling/", model, online)


@flask_app.route("/")
def home_index() -> Any:
    return render_template("index.html")


@flask_app.route("/dash_inference")
def dash_inference_index():
    return dash_inference.index()


@flask_app.route("/dash_labeling")
def dash_labeling_index():
    return dash_labeling.index()


if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port="8000", debug=True)
