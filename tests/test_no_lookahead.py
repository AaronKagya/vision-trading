"""Check chart rendering and inference have no access to later candles."""
import numpy as np
import pytest
import torch
from src.data import synthetic
from src.features import panels, render, numeric_features, first_valid_index
from src.models import CNNEncoder, WorldModel, _img, predict_world_model


def test_future_candles_cannot_change_past_chart_or_numbers():
    df = synthetic(650, seed=3)
    idx, p = panels(df, 32)
    timestamp = idx[70]
    original_img = render(p[70:71], 64)
    original_num = numeric_features(p[70:71])
    changed = df.copy()
    for col in ['open', 'high', 'low', 'close']:
        changed.loc[timestamp + 1:, col] *= 2.5
    j, pp = panels(changed, 32)
    np.testing.assert_array_equal(idx, j)
    np.testing.assert_array_equal(original_img, render(pp[70:71], 64))
    np.testing.assert_array_equal(original_num, numeric_features(pp[70:71]))
    _, truncated = panels(df.iloc[:timestamp + 1], 32)
    np.testing.assert_array_equal(original_img, render(truncated[-1:], 64))


def test_panels_use_one_shared_scale_per_timeframe():
    df = synthetic(650, seed=1)
    _, pan = panels(df, 32)
    img = render(pan[:1], 64)
    assert img.shape == (1, 3, 64, 64)
    assert img.max() == 255
    # Up/down bodies occupy multiple columns, and wicks exist in chart data.
    assert (img > 0).sum() > 100


def test_world_model_rejects_future_frame_inputs():
    model = WorldModel(CNNEncoder(), mode='wm', K=3)
    with pytest.raises(ValueError, match='historical frames'):
        model(torch.randn(2, 4, 3, 64, 64))
    with torch.no_grad():
        logit, future_hat, z = model(torch.randn(2, 3, 3, 64, 64))
    assert logit.shape == (2,)
    assert future_hat.shape == (2, 64)
    assert z.shape == (2, 3, 64)


def test_predictions_ignore_unseen_future_images():
    rng = np.random.default_rng(4)
    imgs = rng.integers(0, 256, size=(8, 3, 64, 64), dtype=np.uint8)
    model = WorldModel(CNNEncoder(), mode='wm', K=3)
    indices = np.array([3, 4])
    offs = np.arange(-2, 1)
    p1, _ = predict_world_model(model, offs, imgs, indices, bs=2)
    changed = imgs.copy()
    changed[5:] = 0  # no effect on predictions for t=3 or t=4
    p2, _ = predict_world_model(model, offs, changed, indices, bs=2)
    np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-7)
    with pytest.raises(ValueError, match='future frame'):
        predict_world_model(model, np.array([-1, 0, 1]), imgs, indices)


def test_model_image_index_bounds():
    imgs = np.zeros((2, 3, 64, 64), dtype=np.uint8)
    with pytest.raises(IndexError):
        _img(imgs, [2])
