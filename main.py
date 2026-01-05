import numpy as np
import matplotlib.pyplot as plt


class LevelSet:
    def __init__(self):
        pass

    def get_value(self, *coords):
        pass


class SphereLevelSet(LevelSet):
    def __init__(self, radius, origin=None):
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


class Plot2DContourf:
    def __init__(
        self,
        x,
        y,
        data,
        x_label: str = None,
        y_label: str = None,
        c_bar_label: str = None,
    ):
        # TODO: Add assertion for size mismatch
        self.fig, self.ax = plt.subplots()
        self.contour = self.ax.contourf(x, y, data)

        if x_label is not None:
            self.fig.xlabel(x_label)
        if y_label is not None:
            self.fig.ylabel(y_label)

        self.cbar = self.fig.colorbar(self.contour, ax=self.ax)
        if c_bar_label is not None:
            self.cbar.set_label(c_bar_label)

    def save(self, filename: str):
        self.fig.savefig(filename, dpi=300)


positions = np.linspace(0, 100, 100)

x, y = np.meshgrid(positions, positions)

sphere_level_set = SphereLevelSet(40, [50, 50])
data = sphere_level_set.get_value(x, y)

plane_level_set = PlaneLevelSet([0, 1], [40, 40])
data = plane_level_set.get_value(x, y)

plotter = Plot2DContourf(x, y, data)
plotter.save("contour.png")
