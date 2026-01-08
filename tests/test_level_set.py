import numpy as np
import pytest

from tpb_measurement import level_set


def test_base_invalid_dimension():
    base = level_set.LevelSet()
    with pytest.raises(ValueError):
        base.get_value(0, 0, 0, 0)


def test_base_mismatch_array_types():
    base = level_set.LevelSet()
    with pytest.raises(ValueError):
        x = np.zeros((3, 1))
        y = 0
        base.get_value(x, y)


def test_base_mismatch_array_shapes():
    base = level_set.LevelSet()
    with pytest.raises(ValueError):
        x = np.zeros((3, 1))
        y = np.zeros((4, 1))
        base.get_value(x, y)


def test_base_flat_array():
    base = level_set.LevelSet()
    with pytest.raises(ValueError):
        x = np.zeros((3, 2))
        base.get_value(x)
