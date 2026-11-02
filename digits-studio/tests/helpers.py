"""Générateurs de données de test : nombres dessinés et fausses photos.

Les chiffres sont piochés uniquement dans le jeu de TEST de Digits (jamais vus à l'entraînement).
"""
import io

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

_D = load_digits()
_, TEST_IDX = train_test_split(
    np.arange(len(_D.target)), test_size=0.2, stratify=_D.target, random_state=42
)

PHOTO_KINDS = ["shadow", "ruled", "thin", "small", "rotated", "inverted"]


def draw_number(s, rng, touching=False):
    """Nombre `s` dessiné sur un canvas 320x960 (blanc sur noir), float [0, 1]."""
    parts = []
    for ch in s:
        idx = rng.choice(TEST_IDX[_D.target[TEST_IDX] == int(ch)])
        big = ndi.zoom(_D.images[idx] / 16, 14, order=3)
        big = np.clip(ndi.gaussian_filter(big, 2) * 2.2, 0, 1)
        big = ndi.grey_dilation(big, size=(6, 6))
        gap = np.zeros((big.shape[0], 0 if touching else 40))
        parts += [big, gap]
    row = np.hstack(parts)
    canvas = np.zeros((320, 960))
    y = (320 - row.shape[0]) // 2
    canvas[y:y + row.shape[0], 20:20 + row.shape[1]] = row[:, :940]
    return canvas


def canvas_png(s, rng, touching=False):
    """Même chose, encodé en PNG (comme l'envoie le navigateur)."""
    buf = io.BytesIO()
    Image.fromarray((draw_number(s, rng, touching) * 255).astype(np.uint8)).save(buf, "PNG")
    buf.seek(0)
    return buf


def _ink_layer(s, rng, thin=False):
    img = draw_number(s, rng)
    if thin:
        img = ndi.grey_erosion(img, size=(7, 7))
    ys, xs = np.where(img > 0.1)
    return img[ys.min() - 10: ys.max() + 10, xs.min() - 10: xs.max() + 10]


def make_photo(s, kind, rng):
    """Fausse photo JPEG (qualité 55) du nombre `s` : ombre, bruit, lignes, rotation..."""
    ink = _ink_layer(s, rng, thin=(kind == "thin"))
    H, W = 900, 1400
    page = np.zeros((H, W))
    ink = ndi.zoom(ink, 0.45 if kind == "small" else 1.0, order=1)
    h, w = ink.shape
    y = rng.integers(20, max(21, H - h - 20))
    x = rng.integers(20, max(21, W - w - 20))
    page[y:y + h, x:x + w] = ink[:H - y, :W - x]
    shade = np.linspace(rng.uniform(0.55, 0.8), 1.0, W)[None, :] * np.linspace(0.85, 1.0, H)[:, None]
    gray = 0.88 * shade * (1 - 0.85 * page)
    if kind in ("ruled", "small"):
        for yy in range(60, H, 70):
            gray[yy:yy + 2, :] *= 0.6
        gray[:, 90:92] *= 0.7
    gray += rng.normal(0, 0.02, gray.shape)
    gray = ndi.gaussian_filter(gray, 1.2)
    if kind == "rotated":
        gray = ndi.rotate(gray, rng.uniform(-6, 6), reshape=False, order=1, cval=0.8)
    gray = np.clip(gray, 0, 1)
    if kind == "inverted":
        gray = 1 - gray                      # tableau noir, craie blanche
    buf = io.BytesIO()
    Image.fromarray((gray * 255).astype(np.uint8)).convert("RGB").save(buf, "JPEG", quality=55)
    buf.seek(0)
    return buf


def random_number(rng):
    length = rng.integers(2, 6)
    return str(rng.integers(1, 10)) + "".join(map(str, rng.integers(0, 10, length - 1)))
