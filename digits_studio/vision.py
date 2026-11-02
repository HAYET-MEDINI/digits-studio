"""Vision : du dessin brut (canvas) à une liste de chiffres 8x8 prêts pour le MLP."""
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi


def _runs(mask_1d):
    """Intervalles [a, b[ de True consécutifs."""
    padded = np.concatenate([[0], mask_1d.astype(int), [0]])
    d = np.diff(padded)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _split_wide(col_profile, a, b, height, depth=0):
    """Si un segment est trop large pour un seul chiffre (chiffres collés),
    on le coupe à la vallée de la projection verticale."""
    w = b - a
    if w / max(height, 1) <= 1.15 or w < 8 or depth > 4:
        return [(a, b)]
    lo, hi = a + int(0.3 * w), b - int(0.3 * w)
    if hi <= lo:
        return [(a, b)]
    cut = lo + int(np.argmin(col_profile[lo:hi]))
    return (_split_wide(col_profile, a, cut, height, depth + 1)
            + _split_wide(col_profile, cut, b, height, depth + 1))


def segment(img, ink_thr=0.15):
    """img : tableau float [0,1], encre = valeurs hautes.
    Retourne la liste des boîtes (x0, x1, y0, y1), de gauche à droite."""
    mask = img > ink_thr
    if mask.sum() < 20:
        return []
    ys, xs = np.where(mask)
    total_h = ys.max() - ys.min() + 1
    col_ink = mask.sum(axis=0)
    min_gap = max(3, int(0.06 * total_h))
    # on comble les petits trous internes à un chiffre
    closed = ndi.binary_closing(mask.any(axis=0), structure=np.ones(min_gap))
    boxes = []
    for a, b in _runs(closed):
        if col_ink[a:b].sum() < 15:          # bruit
            continue
        sub = mask[:, a:b]
        rows = np.where(sub.any(axis=1))[0]
        h = rows.max() - rows.min() + 1
        for s, e in _split_wide(col_ink, a, b, h):
            part = mask[:, s:e]
            r = np.where(part.any(axis=1))[0]
            if len(r) and part.sum() >= 15:
                boxes.append((int(s), int(e), int(r.min()), int(r.max()) + 1))
    return boxes


def stroke_width(mask):
    """Épaisseur moyenne du trait : 2 x aire / périmètre."""
    perim = (mask & ~ndi.binary_erosion(mask)).sum()
    return 2.0 * mask.sum() / max(perim, 1)


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def normalize_stroke(crop, target):
    """Ramène l'épaisseur du trait vers `target` x hauteur (stylo fin ou gros feutre)."""
    size = max(crop.shape)
    m = crop > 0.4
    if m.sum() < 10:
        return crop
    w, want = stroke_width(m), target * size
    if w < 0.9 * want:
        r = min(int(round((want - w) / 2)), max(1, int(0.08 * size)))
        if r >= 1:
            crop = ndi.grey_dilation(crop, footprint=_disk(r))
    elif w > 1.5 * want:
        r = min(int(round((w - want) / 2)), max(0, int(w / 2) - 1))
        if r >= 1:
            crop = ndi.grey_erosion(crop, footprint=_disk(r))
    return crop


def to_8x8(img, box, margin=0.12, target_stroke=None):
    """Recadre, rend carré, recentre sur le centre de masse, redimensionne en 8x8 (0-16)."""
    x0, x1, y0, y1 = box
    crop = img[y0:y1, x0:x1]
    if target_stroke:
        crop = normalize_stroke(crop, target_stroke)
    h, w = crop.shape
    side = int(max(h, w) * (1 + 2 * margin))
    canvas = np.zeros((side, side), dtype=float)
    cy, cx = ndi.center_of_mass(crop)
    oy = int(np.clip(side / 2 - cy, 0, side - h))
    ox = int(np.clip(side / 2 - cx, 0, side - w))
    canvas[oy:oy + h, ox:ox + w] = crop
    small = Image.fromarray((canvas * 255).astype(np.uint8)).resize((8, 8), Image.BOX)
    arr = np.asarray(small, dtype=float)
    if arr.max() > 0:
        arr = arr / arr.max() * 16
    return arr


def read_canvas(pil_img):
    """PIL (canvas blanc sur noir) -> img float [0,1]."""
    g = np.asarray(pil_img.convert("L"), dtype=float) / 255.0
    return g


# ---------------------------------------------------------------------------
#  Photos / scans : n'importe quelle image contenant un nombre manuscrit
# ---------------------------------------------------------------------------
def _otsu(x):
    hist, _ = np.histogram(x, bins=256, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    w, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    sb = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return (int(np.argmax(sb)) + 0.5) / 256


def load_photo(pil, max_side=1200, min_side=500):
    """Ouvre n'importe quelle image : orientation EXIF, transparence, taille raisonnable."""
    pil = ImageOps.exif_transpose(pil)
    if pil.mode in ("RGBA", "LA", "P"):
        rgba = pil.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        pil = bg
    pil = pil.convert("L")
    w, h = pil.size
    m = max(w, h)
    target = max_side if m > max_side else (min_side if m < min_side else m)
    if target != m:
        k = target / m
        pil = pil.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
    return np.asarray(pil, dtype=float) / 255.0


def remove_rules(mask, frac=0.8):
    """Efface les lignes de cahier / marges : segments rectilignes couvrant plus de `frac` de la page
    (les petites interruptions sont comblées). Garde-fou : si presque rien ne reste après
    suppression (ex. un '1' seul, bien cadré), on ne touche à rien."""
    h, w = mask.shape
    out = mask.copy()
    band = np.zeros_like(mask)
    for gap, limit in ((np.ones((1, 7), bool), max(25, int(frac * w))),
                       (np.ones((7, 1), bool), max(25, int(frac * h)))):
        joined = ndi.binary_closing(mask, structure=gap)
        struct = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]) if gap.shape[0] == 1 \
            else np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        lab, n = ndi.label(joined, structure=struct)
        if n == 0:
            continue
        sizes = ndi.sum(joined, lab, range(1, n + 1))
        long_run = np.concatenate([[False], sizes >= limit])[lab]
        cut = ndi.binary_dilation(long_run, iterations=2)
        band |= cut
        out &= ~cut
    # répare les chiffres traversés par une ligne : on recolle seulement dans la bande supprimée,
    # là où il y a de l'encre de part et d'autre
    for fill in (np.ones((9, 1), bool), np.ones((1, 9), bool)):
        out |= ndi.binary_closing(out, structure=fill) & band & mask
    return out if out.sum() >= max(30, 0.02 * mask.sum()) else mask


def extract_ink(gray, polarity="auto"):
    """Gris -> carte d'encre [0,1] (encre = valeurs hautes), robuste à l'éclairage,
    à la couleur du papier, aux ombres, aux lignes de cahier et au bruit.
    polarity : 'auto', 'dark' (encre foncée) ou 'light' (encre claire)."""
    h, w = gray.shape
    med = np.median(gray)
    if polarity == "auto":
        dev_dark = med - np.percentile(gray, 0.2)
        dev_light = np.percentile(gray, 99.8) - med
        dark_ink = dev_dark >= dev_light
    else:
        dark_ink = polarity == "dark"

    # estimation du fond (papier + ombres) sur une version réduite
    f = max(1, max(h, w) // 100)
    small = ndi.zoom(gray, 1 / f, order=1)
    k = max(9, int(0.3 * max(small.shape)) | 1)
    q = 85 if dark_ink else 15
    bg_s = ndi.percentile_filter(small, q, size=k, mode="nearest")
    bg = ndi.zoom(bg_s, (h / bg_s.shape[0], w / bg_s.shape[1]), order=1)[:h, :w]

    ink = np.clip((bg - gray) if dark_ink else (gray - bg), 0, None)
    scale = np.percentile(ink, 99.5)
    if scale < 0.10:                       # page vide
        return np.zeros_like(gray)
    ink = np.clip(ink / max(ink.max() * 0.6, scale), 0, 1)
    mask = ink > max(_otsu(ink), 0.2)

    # nettoyage : lignes de cahier / marges, puis points isolés
    mask = remove_rules(mask)
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(gray)
    boxes = ndi.find_objects(lab)
    dims = np.array([[b[0].stop - b[0].start, b[1].stop - b[1].start] for b in boxes])
    areas = ndi.sum(mask, lab, range(1, n + 1))
    ratio = dims[:, 1] / np.maximum(dims[:, 0], 1)
    ruled = ratio > 15                                          # lignes de cahier
    margin = (dims[:, 0] > 0.85 * h) & (ratio < 0.01)           # traits de marge
    # blocs pleins et très grands (aplats, bords de table) : critère volontairement strict, car un chiffre
    # tracé au gros feutre remplit déjà beaucoup sa boîte ; les ombres grises sont gérées par le contraste
    boxarea = dims[:, 0] * dims[:, 1]
    blob = (areas / np.maximum(boxarea, 1) > 0.85) & (boxarea > 0.05 * h * w)
    line = ruled | margin | (blob if not blob.all() else np.zeros(n, bool))
    pool = ~line if (~line).any() else np.ones(n, bool)
    big = dims[pool].max()                                      # taille de référence : sans les lignes
    amax = areas[pool].max()
    tiny = (dims.max(axis=1) < 0.12 * big) | (areas < 0.01 * amax)
    # contraste relatif : l'encre est bien plus contrastée que les zones grises du décor
    peaks = np.asarray(ndi.maximum(ink, lab, range(1, n + 1)))
    ref_pool = ~line & ~tiny
    ref = peaks[ref_pool].max() if ref_pool.any() else peaks.max()
    weak = peaks < 0.7 * ref
    keep = np.concatenate([[False], ~(tiny | line | weak)])
    clean = keep[lab]
    clean = ndi.binary_dilation(clean, iterations=1)
    return ink * clean


def stroke_variants(img, box):
    """Plusieurs lectures du même chiffre (marge / épaisseur du trait) : on moyenne les prédictions."""
    return np.array([to_8x8(img, box, margin=m, target_stroke=t)
                     for m, t in ((0.10, None), (0.14, None), (0.12, 0.12), (0.12, 0.16))])
