"""Digits Studio : lecture de nombres manuscrits (MLP scikit-learn + Flask)."""
import os
from pathlib import Path

import joblib
from flask import Flask

__version__ = "1.0.0"

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("DIGITS_MODEL", ROOT / "models" / "model.joblib"))


def create_app(model_path=None):
    """Fabrique de l'application Flask (le modèle est chargé une seule fois)."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 Mo
    path = Path(model_path or MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {path}\n"
            "Entraînez-le avec :  python -m digits_studio.training"
        )
    app.config["MODEL"] = joblib.load(path)

    from .web import bp
    app.register_blueprint(bp)
    return app
