import numpy as np

from tpb_measurement import numerics


def test_1d_normal():
    dim = 1
    n = 10
    forward = np.linspace(0, 1, round(n / 2))
    backward = np.linspace(1, 0, round(n / 2))
    data = np.append(forward, backward)

    normal = numerics.NormalData(data)
    normal_data = normal.get_normal()

    expected_normal_data = np.ones_like(normal_data)
    expected_normal_data[0] = 0
    expected_normal_data[-1] = 0

    assert normal_data.shape == (n, dim)
    assert (normal_data == expected_normal_data).all()


def test_2d_normal():
    dim = 2
    n = 10
    forward = np.linspace(0, 1, round(n / 2))
    backward = np.linspace(1, 0, round(n / 2))
    data = np.append(forward, backward)
    data = np.tile(data, n ** (dim - 1))
    data = np.reshape(data, (n, n))

    normal = numerics.NormalData(data)
    normal_data = normal.get_normal()

    expected_normal_data = np.ones_like(normal_data)
    expected_normal_data[:, 0] = 0
    expected_normal_data[:, -1] = 0
    expected_normal_data[..., 1] = 0

    assert normal_data.shape == (n, n, dim)
    assert (normal_data == expected_normal_data).all()


def test_3d_normal():
    dim = 3
    n = 10
    forward = np.linspace(0, 1, round(n / 2))
    backward = np.linspace(1, 0, round(n / 2))
    data = np.append(forward, backward)
    data = np.tile(data, n ** (dim - 1))
    data = np.reshape(data, (n, n, n))

    normal = numerics.NormalData(data)
    normal_data = normal.get_normal()

    expected_normal_data = np.ones_like(normal_data)
    expected_normal_data[:, :, 0] = 0
    expected_normal_data[:, :, -1] = 0
    expected_normal_data[..., 1] = 0
    expected_normal_data[..., 2] = 0

    assert normal_data.shape == (n, n, n, dim)
    assert (normal_data == expected_normal_data).all()
