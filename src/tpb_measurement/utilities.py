import numpy as np


def assert_shape(v_1: np.ndarray, v_2: np.ndarray):
    assert (
        v_1.shape == v_2.shape
    ), f"Shape mismatch: {v_1.shape} vs {v_2.shape}"


def assert_range_exclusive(v, v_min, v_max):
    if isinstance(v, np.ndarray):
        assert np.all(
            (v > v_min) & (v < v_max)
        ), f"Array values out of range ({v_min}, {v_max})"
    elif isinstance(v, (float, int)):
        assert v_min < v < v_max, f"Value {v} out of range ({v_min}, {v_max})"
    else:
        assert False, f"Unexpected type: {type(v)}"


def assert_range_inclusive(v, v_min, v_max):
    if isinstance(v, np.ndarray):
        assert np.all(
            (v >= v_min) & (v <= v_max)
        ), f"Array values out of range [{v_min}, {v_max}]"
    elif isinstance(v, (float, int)):
        assert (
            v_min <= v <= v_max
        ), f"Value {v} out of range [{v_min}, {v_max}]"
    else:
        assert False, f"Unexpected type: {type(v)}"


def assert_range_with_clipping(v, v_min, v_max, tol: float = 1e-6):
    if isinstance(v, np.ndarray):
        assert np.all(
            v >= (v_min - tol)
        ), f"Array values below {v_min} beyond tolerance {tol}"
        assert np.all(
            v <= (v_max + tol)
        ), f"Array values above {v_max} beyond tolerance {tol}"
        return np.clip(v, v_min, v_max)
    elif isinstance(v, (float, int)):
        assert v >= (
            v_min - tol
        ), f"Value {v} below {v_min} beyond tolerance {tol}"
        assert v <= (
            v_max + tol
        ), f"Value {v} above {v_max} beyond tolerance {tol}"
        return min(max(v, v_min), v_max)
    else:
        assert False, f"Unexpected type: {type(v)}"
