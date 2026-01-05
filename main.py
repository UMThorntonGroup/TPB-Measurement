import numpy as np
import matplotlib.pyplot as plt


class LevelSet:
    def __init__(self):
        pass

    def get_value(self, x, y=None, z=None):
        # TODO: Throw an error if the base class is called
        # TODO: Add types
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

plotter = Plot2DContourf(x, y, data)
plotter.save("contour.png")
