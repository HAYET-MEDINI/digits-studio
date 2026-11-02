import base64
import io

import numpy as np
from PIL import Image

from digits_studio import create_app
from tests.helpers import canvas_png, make_photo, random_number

client = create_app().test_client()


def _post_image(buf, name="x.png", **form):
    return client.post("/predict_image", data={"file": (buf, name), **form},
                       content_type="multipart/form-data")


def test_index_and_health():
    assert client.get("/").status_code == 200
    assert client.get("/health").get_json() == {"status": "ok"}


def test_predict_image_without_file():
    r = client.post("/predict_image", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_predict_image_invalid_file():
    r = _post_image(io.BytesIO(b"pas une image"), "x.jpg")
    assert r.status_code == 400 and "illisible" in r.get_json()["error"]


def test_predict_image_blank_page():
    buf = io.BytesIO()
    Image.new("RGB", (800, 500), (240, 240, 235)).save(buf, "PNG")
    buf.seek(0)
    res = _post_image(buf, "blanc.png").get_json()
    assert res["number"] == "" and "error" in res


def test_reads_drawn_numbers():
    rng = np.random.default_rng(0)
    ok, total = 0, 12
    for _ in range(total):
        s = random_number(rng)
        buf = canvas_png(s, rng)
        url = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
        ok += client.post("/predict", json={"image": url}).get_json()["number"] == s
    assert ok >= 9, f"{ok}/{total} nombres lus"


def test_reads_photos():
    rng = np.random.default_rng(1)
    ok, total = 0, 0
    for kind in ("shadow", "thin", "inverted"):
        for _ in range(4):
            s = random_number(rng)
            res = _post_image(make_photo(s, kind, rng), "p.jpg").get_json()
            ok += res["number"] == s
            total += 1
    assert ok >= 9, f"{ok}/{total} photos lues"
