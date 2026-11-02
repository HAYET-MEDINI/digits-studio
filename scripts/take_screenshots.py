"""Génère les captures d'écran du README en pilotant l'application avec Chromium (Playwright).

Prérequis :  pip install playwright && playwright install chromium
Usage     :  python -m scripts.take_screenshots
Les tracés à la souris sont simulés (polylignes), la lecture est faite par le vrai modèle.
"""
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from tests.helpers import make_photo

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
URL = "http://127.0.0.1:5000"


def _ellipse(n=28):
    pts = [(0.5 + 0.4 * math.cos(a), 0.5 + 0.5 * math.sin(a))
           for a in np.linspace(-math.pi / 2, 1.5 * math.pi + 0.25, n)]
    return pts


# chiffres tracés à la main (coordonnées normalisées dans une boîte 1 x 1)
STROKES = {
    "2": [(0.1, 0.25), (0.25, 0.08), (0.55, 0.02), (0.8, 0.15), (0.82, 0.38), (0.6, 0.62), (0.1, 0.98), (0.92, 0.98)],
    "0": _ellipse(),
    "6": [(0.8, 0.05), (0.5, 0.2), (0.25, 0.5), (0.15, 0.75), (0.25, 0.95), (0.5, 1.0), (0.78, 0.88),
          (0.8, 0.68), (0.55, 0.55), (0.25, 0.65)],
}


def draw_number(page, text, height=0.62, width_ratio=0.5, gap=0.55):
    box = page.locator("#pad").bounding_box()
    h = box["height"] * height
    w = h * width_ratio
    total = len(text) * w + (len(text) - 1) * w * gap
    x0 = box["x"] + (box["width"] - total) / 2
    y0 = box["y"] + (box["height"] - h) / 2
    for i, ch in enumerate(text):
        ox = x0 + i * w * (1 + gap)
        pts = [(ox + px * w, y0 + py * h) for px, py in STROKES[ch]]
        page.mouse.move(*pts[0])
        page.mouse.down()
        for p in pts[1:]:
            page.mouse.move(*p, steps=4)
        page.mouse.up()
        page.wait_for_timeout(120)


def wait_server():
    for _ in range(60):
        try:
            urllib.request.urlopen(URL + "/health", timeout=1)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Le serveur ne démarre pas")


def main():
    DOCS.mkdir(exist_ok=True)
    server = subprocess.Popen([sys.executable, "run.py"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_server()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1000, "height": 900}, device_scale_factor=1.5)

            # 1) dessin à la souris
            page.goto(URL)
            draw_number(page, "2026")
            page.wait_for_function("document.querySelector('#number').textContent !== '—'", timeout=15000)
            page.wait_for_timeout(400)
            print("Dessin lu :", page.inner_text("#number"), "|", page.inner_text("#words"))
            page.screenshot(path=str(DOCS / "screenshot-dessin.png"), full_page=True)

            # 2) import d'une photo (papier ligné, ombre, JPEG compressé)
            photo = DOCS / "exemple-photo.jpg"
            photo.write_bytes(make_photo("4052", "ruled", np.random.default_rng(11)).read())
            page.goto(URL)
            page.click("#tabImg")
            page.set_input_files("#file", str(photo))
            page.wait_for_function("document.querySelector('#number').textContent !== '—'", timeout=20000)
            page.wait_for_timeout(500)
            print("Photo lue :", page.inner_text("#number"), "|", page.inner_text("#words"))
            page.screenshot(path=str(DOCS / "screenshot-import.png"), full_page=True)
            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
