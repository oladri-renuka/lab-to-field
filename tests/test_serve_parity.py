import numpy as np
from PIL import Image

from serve.predictor import preprocess
from src.data.io import load_image
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, eval_transform


def test_serving_preprocess_matches_eval_transform(tmp_path):
    """The ONNX app must see exactly the tensor the model was validated on (resize, EXIF, normalisation)."""
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))
    exif = img.getexif()
    exif[0x0112] = 6  # phone photo stored rotated
    img.save(tmp_path / "leaf.jpg", exif=exif)

    train_side = eval_transform(224, IMAGENET_MEAN, IMAGENET_STD)(load_image(tmp_path / "leaf.jpg")).numpy()[None]
    with Image.open(tmp_path / "leaf.jpg") as raw:  # what an upload looks like before any EXIF handling
        served = preprocess(raw, 224, IMAGENET_MEAN, IMAGENET_STD)
    assert served.shape == train_side.shape == (1, 3, 224, 224)
    np.testing.assert_allclose(served, train_side, atol=1e-5)
