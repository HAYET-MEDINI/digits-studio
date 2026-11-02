"""Benchmark de lecture sur fausses photos difficiles.

Usage :  python -m scripts.benchmark_photos [n_par_cas] [cas ...]
Cas : shadow ruled thin small rotated inverted
"""
import sys

import numpy as np

from digits_studio import create_app
from tests.helpers import PHOTO_KINDS, make_photo, random_number


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 12
    kinds = [a for a in sys.argv[1:] if a in PHOTO_KINDS] or PHOTO_KINDS
    client = create_app().test_client()
    rng = np.random.default_rng(3)
    for kind in kinds:
        ok = 0
        for _ in range(n):
            s = random_number(rng)
            res = client.post("/predict_image", data={"file": (make_photo(s, kind, rng), "p.jpg")},
                              content_type="multipart/form-data").get_json()
            ok += res.get("number") == s
        print(f"{kind:9s}: {ok}/{n} nombres exacts")


if __name__ == "__main__":
    main()
