import numpy as np


class LevelSet:
    def __init__(self):
        pass

    def get_value(self, *coords):
        # Check that the coords have a valid type and dimensions
        dim = len(coords)
        if dim > 3:
            raise ValueError("LevelSet must have at most 3 dimensions")

        array_type = type(coords[0])
        for i in range(dim):
            local_array_type = type(coords[i])
            if local_array_type != array_type:
                raise ValueError("Mismatch in coordinate array types")

        array_length = np.shape(coords[0])
        for i in range(dim):
            local_array_length = np.shape(coords[i])
            if array_length != local_array_length:
                raise ValueError("Mismatch in coordinate array shapes")

        if np.sum(array_length) - 1 != np.prod(array_length):
            raise ValueError("Coordinate array must be a flat vector")


class SphereLevelSet(LevelSet):
    def __init__(self, radius, origin=None):
        super().__init__()
        self.radius = radius

        if origin is None:
            self.origin = None
        else:
            self.origin = np.asarray(origin, dtype=float)
            if self.origin.ndim != 1:
                raise ValueError("Origin must be 1D vector")

    def get_value(self, *coords):
        dim = len(coords)

        if self.origin is None:
            origin = np.zeros(dim)
        else:
            if len(self.origin) != dim:
                raise ValueError(
                    f"Origin dimension {len(self.origin)} does not match "
                    f"input dimension {dim}"
                )
            origin = self.origin

        radius_squared = 0.0
        for coord, origin in zip(coords, origin):
            radius_squared += (coord - origin) ** 2

        return np.sqrt(radius_squared) - self.radius


class PlaneLevelSet(LevelSet):
    def __init__(self, normal, origin=None):
        super().__init__()
        self.normal = np.asarray(normal, dtype=float)
        if self.normal.ndim != 1:
            raise ValueError("normal must be a 1D array-like")

        norm = np.linalg.norm(self.normal)
        if norm == 0:
            raise ValueError("normal vector must be non-zero")
        self.normal /= norm

        if origin is None:
            self.origin = None
        else:
            self.origin = np.asarray(origin, dtype=float)
            if self.origin.ndim != 1:
                raise ValueError("Origin must be 1D vector")

    def get_value(self, *coords):
        dim = len(coords)

        if self.origin is None:
            origin = np.zeros(dim)
        else:
            if len(self.origin) != dim:
                raise ValueError(
                    f"Origin dimension {len(self.origin)} does not match "
                    f"input dimension {dim}"
                )
            origin = self.origin

        x = np.stack(coords, axis=0)

        return np.tensordot(self.normal, x - origin[:, None, None], axes=1)
