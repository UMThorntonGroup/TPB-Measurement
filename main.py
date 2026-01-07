import numpy as np
import matplotlib.pyplot as plt
import math


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


class Plot2DTPB:
    def __init__(self, x, y, data_1, data_2, data_3=None):
        if data_3 is None:
            data_3 = np.ones_like(data_1) - data_1 - data_2

        if not np.allclose(data_1 + data_2 + data_3, np.ones_like(data_1)):
            raise ValueError("The provided data must sum to 1")
        if np.any((data_1 < 0) | (data_2 < 0) | (data_3 < 0)):
            raise ValueError("The provided data must be non-negative")

        self.fig, self.ax = plt.subplots()
        self.ax.contour(x, y, data_1, levels=[0.5], colors="r", linewidths=2)
        self.ax.contour(x, y, data_2, levels=[0.5], colors="g", linewidths=2)
        self.ax.contour(x, y, data_3, levels=[0.5], colors="b", linewidths=2)

        # Identify the TPB points where all three phases are present
        min = 0.1
        is_TPB = (data_1 > min) & (data_2 > min) & (data_3 > min)

        combined = data_1 * data_2 * data_3
        self.ax.imshow(
            is_TPB,
            origin="lower",
            extent=(x.min(), x.max(), y.min(), y.max()),
            cmap="gray",
            alpha=0.3,
            interpolation="nearest",
        )
        self.ax.contourf(x, y, combined)
        self.ax.set_ylim([15, 25])
        self.ax.set_xlim([20, 40])

    def save(self, filename: str):
        self.fig.savefig(filename, dpi=300)


def level_set_to_tanh(data, interfacial_width):
    """
    Convert a level-set field to a tanh ranging from 0 to 1
    """
    return 0.5 * (1 - np.tanh(data / (np.sqrt(2) * interfacial_width)))


def find_positions_from_contact_angle(contact_angle: float, radius: float):
    """
    This is a little helper function that helps find the
    positions a plane and sphere level-set given some contact
    angle.

    For now, it assumes that the domain size is 100 by 100 with
    100 points in either direction.
    """
    x_size = 100
    y_size = 100

    h = 1

    n_points_x = math.ceil(x_size / h)
    n_points_y = math.ceil(y_size / h)

    x_positions = np.linspace(0, x_size, n_points_x)
    y_positions = np.linspace(0, y_size, n_points_y)

    # Create the meshgrid
    x, y = np.meshgrid(x_positions, y_positions)

    # Place the plane level-set somewhere in domain. For now,
    # we'll always place it at y = 20
    plane_y_position = 20
    plane_level_set = PlaneLevelSet([0, 1], [plane_y_position, plane_y_position])
    plane_level_set_data = plane_level_set.get_value(x, y)

    # Compute the origin of the sphere, given the contact angle
    sphere_x_position = x_size / 2

    def compute_sphere_y_position(_radius: float):
        return plane_y_position - _radius * math.cos(contact_angle)

    sphere_y_position = compute_sphere_y_position(radius)

    # Place the sphere level-set
    sphere_level_set = SphereLevelSet(radius, [sphere_x_position, sphere_y_position])
    sphere_level_set_data = sphere_level_set.get_value(x, y)

    # Note that this constructs the full sphere, however to simulate
    # the triple-boundary we have to cut-off the sphere where it intersects
    # the plane. This is trivial with level-set as we take minimum of the two sets
    # From there, we can calculate the third phase and recompute the sphere phase
    interfacial_width = 1
    combined_level_set = np.minimum(sphere_level_set_data, plane_level_set_data)
    combined_tanh = level_set_to_tanh(combined_level_set, 1)

    # Convert the level-set fields to tanh
    sphere_tanh = level_set_to_tanh(sphere_level_set_data, interfacial_width)
    plane_tanh = level_set_to_tanh(plane_level_set_data, interfacial_width)

    matrix_tanh = 1.0 - combined_tanh
    sphere_tanh = combined_tanh - plane_tanh

    plotter = Plot2DTPB(x, y, plane_tanh, sphere_tanh)
    plotter.save("contour.png")


find_positions_from_contact_angle(math.pi / 2, 20)
