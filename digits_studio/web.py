"""Routes HTTP : interface, lecture du canvas, lecture d'une photo."""
import base64
import io

import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request
from PIL import Image

from .inference import analyse
from .vision import extract_ink, load_photo, read_canvas

bp = Blueprint("web", __name__)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/health")
def health():
    return jsonify(status="ok")


@bp.post("/predict")
def predict():
    """Dessin du canvas (blanc sur noir), envoyé en data-URL PNG."""
    data = request.get_json(force=True)["image"].split(",", 1)[-1]
    pil = Image.open(io.BytesIO(base64.b64decode(data)))
    return jsonify(analyse(current_app.config["MODEL"], read_canvas(pil)))


@bp.post("/predict_image")
def predict_image():
    """Photo ou scan quelconque (JPG, PNG, WEBP, BMP, GIF)."""
    f = request.files.get("file")
    if f is None:
        return jsonify(error="Aucun fichier reçu."), 400
    try:
        pil = Image.open(f.stream)
        pil.load()
    except Exception:
        return jsonify(error="Image illisible (formats : JPG, PNG, WEBP, BMP, GIF)."), 400
    ink = extract_ink(load_photo(pil), request.form.get("polarity", "auto"))
    res = analyse(current_app.config["MODEL"], ink, robust=True)
    buf = io.BytesIO()
    Image.fromarray((ink * 255).astype(np.uint8)).save(buf, "PNG")
    res["preview"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    res["size"] = [int(ink.shape[1]), int(ink.shape[0])]
    if not res["digits"]:
        res["error"] = "Aucun chiffre détecté. Essayez l'option de polarité (encre foncée / claire)."
    return jsonify(res)
