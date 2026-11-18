"""Prétraitement : recadrage du capteur 120 DPI, normalisation par image."""
import numpy as np

from prmi.preprocess import crop_for, normalize, preprocess_image


def test_crop_120_removes_padding_bands():
    img = np.zeros((480, 640, 3), np.uint8)
    assert img[crop_for(120)].shape == (468, 616, 3)
    assert img[crop_for(150)].shape == img.shape


def test_normalize_stretches_to_unit_range():
    rng = np.random.default_rng(0)
    img = rng.uniform(0.1, 0.9, (50, 60, 3)).astype(np.float32)
    out, span = normalize(img)
    assert out.min() >= 0 and out.max() <= 1
    assert np.percentile(out, 99) > 0.95 and np.percentile(out, 1) < 0.05


def test_normalize_caps_gain_on_flat_images():
    img = np.full((40, 40, 3), 0.5, np.float32) + np.random.default_rng(0).normal(0, 0.01, (40, 40, 3)).astype(np.float32)
    out, span = normalize(img, max_gain=3)
    assert np.all(span >= 1 / 3 - 1e-6)                # étirement limité à x3 : le bruit n'est pas gonflé
    assert out.std() < 0.05


def test_preprocess_image_shape_and_range():
    rgb = np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    out = preprocess_image(rgb, np.ones((468, 616), np.float32), 120)
    assert out.shape == (468, 616, 3) and out.dtype == np.float32
    assert 0 <= out.min() and out.max() <= 1
