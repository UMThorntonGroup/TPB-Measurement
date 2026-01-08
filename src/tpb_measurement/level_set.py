import numpy as np


class LevelSet:
    def __init__(self, origin=None):
        if origin is None:
            # The none argument will later be used for the default origin of 0
            self.origin = None
        else:
            self.origin = np.asarray(origin, dtype=float)
            if self.origin.ndim != 1:
                raise ValueError("Origin must be a flat vector")
            self.dim = len(self.origin)

    def get_value(self, *coords):
        # Check that the coords have a valid type and dimensions
        dim = len(coords)
        if dim > 3:
            raise ValueError("LevelSet must have at most 3 dimensions")
        if self.origin is not None and self.dim != dim:
            raise ValueError(
                "get_value must be called with the number of"
                "dimensions as the constructor with the origin"
            )

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

        if isinstance(coords[0], np.ndarray) and coords[0].ndim != 1:
            raise ValueError("Coordinate array must be a flat vector")

        if self.origin is None:
            self.origin = np.zeros(dim)


class SphereLevelSet(LevelSet):
    def __init__(self, radius, origin=None):
        super().__init__(origin)
        if radius <= 0:
            raise ValueError("SphereLevelSet radius must be positive")
        self.radius = radius

    def get_value(self, *coords):
        super().get_value(*coords)
        radius_squared = 0.0
        for coord, origin in zip(coords, self.origin):
            radius_squared += (coord - origin) ** 2

        return np.sqrt(radius_squared) - self.radius


class PlaneLevelSet(LevelSet):
    def __init__(self, normal, origin=None):
        super().__init__(origin)
        self.normal = np.asarray(normal, dtype=float)
        if self.normal.ndim != 1:
            raise ValueError("normal vector must be a 1D array-like")

        norm = np.linalg.norm(self.normal)
        if norm == 0:
            raise ValueError("normal vector must be non-zero")
        self.normal /= norm

    def get_value(self, *coords):
        super().get_value(*coords)
        x = np.stack(coords, axis=0)
        origin = self.origin[:, None, None]
        return np.tensordot(self.normal, x - origin, axes=1)
