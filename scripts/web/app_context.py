from flask import Flask


class AppContext(object):
    _app = None
    _model = None
    _online = False

    @classmethod
    def app(cls):
        if cls._app is None:
            cls._app = Flask(__name__)
            import aic.misc.files as fs
            import aic.model.loaders as ld

            config = fs.get_configs_root() / "web_config.yml"
            config = ld.load_config(str(config))
            model_name = config.get("model_name", None)
            online = config.get("online", False)
            if model_name:
                if online:
                    cls._model = ld.load_tflitemodel(model_name)
                    cls._online = True
                else:
                    cls._model = ld.load_model(model_name)
        return cls._app, cls._model, cls._online
