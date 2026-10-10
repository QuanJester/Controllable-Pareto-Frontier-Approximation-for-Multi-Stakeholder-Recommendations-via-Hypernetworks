"""
Unit tests for checking Pareto evaluation metrics (Hypervolume, Spacing, Spread).
"""

import numpy as np
import pytest
from src.objectives import compute_spacing, compute_spread, compute_hypervolume_2d


def test_spacing_uniform_points():
    # Điểm cách đều nhau thì spacing nên xấp xỉ 0
    points = np.array([
        [0.1, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6],
    ])
    sp = compute_spacing(points)
    assert isinstance(sp, float)
    assert sp >= 0.0
    assert sp < 1e-4, f"Spacing should be very small for equidistant points, got {sp}"


def test_spread_metric():
    points = np.array([
        [0.1, 0.9],
        [0.3, 0.7],
        [0.5, 0.5],
        [0.7, 0.3],
        [0.9, 0.1],
    ])
    delta = compute_spread(points)
    assert isinstance(delta, float)
    assert delta >= 0.0


def test_hypervolume_2d():
    ref_point = np.array([1.0, 1.0])
    points = np.array([
        [0.2, 0.8],
        [0.5, 0.5],
        [0.8, 0.2],
    ])
    hv = compute_hypervolume_2d(points, ref_point)
    assert isinstance(hv, float)
    assert hv > 0.0
    assert hv <= 1.0
