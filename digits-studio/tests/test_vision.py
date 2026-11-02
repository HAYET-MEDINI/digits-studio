import numpy as np

from digits_studio.vision import extract_ink, remove_rules, segment, to_8x8
from tests.helpers import draw_number


def test_segment_blank_image():
    assert segment(np.zeros((100, 300))) == []


def test_segment_three_separated_blobs_left_to_right():
    img = np.zeros((100, 300))
    for x in (20, 100, 200):
        img[20:80, x:x + 25] = 1.0
    boxes = segment(img)
    assert len(boxes) == 3
    assert [b[0] for b in boxes] == sorted(b[0] for b in boxes)


def test_to_8x8_shape_and_scale():
    img = np.zeros((100, 100))
    img[10:90, 40:55] = 1.0
    arr = to_8x8(img, (40, 55, 10, 90))
    assert arr.shape == (8, 8)
    assert abs(arr.max() - 16) < 1e-9


def test_extract_ink_blank_page_is_empty():
    assert extract_ink(np.full((300, 400), 0.9)).max() == 0


def test_extract_ink_polarity_auto_dark_and_light():
    canvas = draw_number("4052", np.random.default_rng(0))
    dark_on_light = 0.9 * (1 - 0.9 * canvas)
    light_on_dark = 1 - dark_on_light
    for gray in (dark_on_light, light_on_dark):
        assert len(segment(extract_ink(gray))) == 4


def test_remove_rules_deletes_page_wide_line_but_keeps_digit():
    mask = np.zeros((200, 600), bool)
    mask[100:103, :] = True            # ligne de cahier
    mask[40:120, 280:300] = True       # un "chiffre"
    out = remove_rules(mask)
    assert out.sum() < mask.sum()
    assert out[60:80, 280:300].all()


def test_remove_rules_guard_keeps_lone_vertical_stroke():
    mask = np.zeros((100, 100), bool)
    mask[:, 45:50] = True              # un "1" qui remplit toute l'image
    assert remove_rules(mask).sum() == mask.sum()
