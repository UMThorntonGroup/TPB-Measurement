import numpy as np
import pytest

from tpb_measurement import level_set


def test_base_invalid_dimension():
    base = level_set.LevelSet()
    with pytest.raises(ValueError):
        base.get_value(0, 0, 0, 0)


def test_base_mismatch_array_types():
    base = level_set.LevelSet()
    x = np.zeros(3)
    y = 0
    with pytest.raises(ValueError):
        base.get_value(x, y)


def test_base_mismatch_array_shapes():
    base = level_set.LevelSet()
    x = np.zeros(3)
    y = np.zeros(4)
    with pytest.raises(ValueError):
        base.get_value(x, y)


def test_base_flat_array():
    base = level_set.LevelSet()
    x = np.zeros((3, 1))
    with pytest.raises(ValueError):
        base.get_value(x)


def test_base_default_origin():
    base = level_set.LevelSet()
    base.get_value(0, 0, 0)
    assert (base.origin == np.zeros(3)).all()


def test_base_mismatch_origin_and_array():
    base = level_set.LevelSet([0, 0])
    with pytest.raises(ValueError):
        base.get_value(0, 0, 0)


def test_sphere_negative_radius():
    with pytest.raises(ValueError):
        sphere = level_set.SphereLevelSet(0)  # noqa: F841


def test_sphere_1d():
    sphere = level_set.SphereLevelSet(10, origin=[0])
    x = np.linspace(0, 20, 10)
    values = sphere.get_value(x)
    assert (values == np.linspace(-10, 10, 10)).all()


def test_plane_zero_normal():
    with pytest.raises(ValueError):
        plane = level_set.PlaneLevelSet([0, 0])  # noqa: F841


def test_plane_1d():
    plane = level_set.PlaneLevelSet([1], origin=[10])
    x = np.linspace(0, 20, 10)
    values = plane.get_value(x)
    assert (values == np.linspace(-10, 10, 10)).all()
