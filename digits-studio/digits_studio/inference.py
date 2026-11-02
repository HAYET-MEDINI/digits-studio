"""Lecture d'un nombre : segmentation -> 8x8 -> MLP -> assemblage."""
import numpy as np

from .numbers_fr import to_words
from .vision import segment, stroke_variants, to_8x8


def analyse(model, img, robust=False):
    """img : carte d'encre float [0, 1] (encre = valeurs hautes).
    robust=True : moyenne de plusieurs lectures par chiffre (utile pour les photos)."""
    boxes = segment(img)
    if not boxes:
        return {"number": "", "digits": [], "words": ""}
    digits = []
    for box in boxes:
        views = stroke_variants(img, box) if robust else np.array([to_8x8(img, box)])
        proba = model.predict_proba(views.reshape(len(views), -1)).mean(axis=0)
        order = np.argsort(proba)[::-1]
        digits.append({
            "digit": int(order[0]),
            "confidence": float(proba[order[0]]),
            "alternatives": [{"digit": int(k), "p": float(proba[k])} for k in order[1:3]],
            "box": box,
            "thumb": (to_8x8(img, box) / 16).round(2).tolist(),
        })
    number = "".join(str(d["digit"]) for d in digits)
    words = to_words(int(number)) if not number.startswith("0") or number == "0" else ""
    return {"number": number, "digits": digits, "words": words}
