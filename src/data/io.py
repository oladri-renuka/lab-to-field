"""Single image loader used everywhere: EXIF orientation applied, always 3-channel RGB."""
from __future__ import annotations

from PIL import Image, ImageOps


def to_rgb(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)  # phone photos otherwise load rotated
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return img.convert("RGB")


def load_image(path) -> Image.Image:
    with Image.open(path) as im:
        im.seek(0)  # first frame of GIFs
        return to_rgb(im)
