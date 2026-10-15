import numpy as np
from PIL import Image

from src.data.io import load_image


def test_rgba_and_grayscale_become_rgb(tmp_path):
    rgba = Image.new("RGBA", (10, 10), (0, 255, 0, 0))
    rgba.save(tmp_path / "a.png")
    Image.new("L", (10, 10), 128).save(tmp_path / "g.png")
    a, g = load_image(tmp_path / "a.png"), load_image(tmp_path / "g.png")
    assert a.mode == g.mode == "RGB"
    assert np.asarray(a)[0, 0].tolist() == [255, 255, 255]  # transparent -> white, not black


def test_exif_orientation_applied(tmp_path):
    img = Image.new("RGB", (20, 10))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 CW on display
    img.save(tmp_path / "r.jpg", exif=exif)
    assert load_image(tmp_path / "r.jpg").size == (10, 20)
