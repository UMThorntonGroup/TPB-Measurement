import numpy as np
import vtk.util.numpy_support


class Output:
    def __init__(self):
        pass

    @staticmethod
    def NumpyToVTK(data, filename: str = "test.vtk", time: float = 0.0):
        data_type = vtk.VTK_FLOAT
        shape = data.shape

        # Flatten the data array so we only to to convert the numpy array once
        flat_data_array = data.flatten()
        vtk_data = vtk.util.numpy_support.numpy_to_vtk(
            num_array=flat_data_array, deep=True, array_type=data_type
        )

        # Create the rectilinear grid object
        grid = vtk.vtkRectilinearGrid()

        grid.GetPointData().SetScalars(vtk_data)

        if len(shape) == 3:
            grid.SetDimensions(shape[0], shape[1], shape[2])

            x = np.linspace(0, 100, shape[0])
            y = np.linspace(0, 100, shape[1])
            z = np.linspace(0, 100, shape[2])

            vtk_x = vtk.util.numpy_support.numpy_to_vtk(
                num_array=x, deep=True, array_type=data_type
            )
            vtk_y = vtk.util.numpy_support.numpy_to_vtk(
                num_array=y, deep=True, array_type=data_type
            )
            vtk_z = vtk.util.numpy_support.numpy_to_vtk(
                num_array=z, deep=True, array_type=data_type
            )

            grid.SetXCoordinates(vtk_x)
            grid.SetYCoordinates(vtk_y)
            grid.SetZCoordinates(vtk_z)

        elif len(shape) == 2:
            grid.SetDimensions(shape[0], shape[1], 1)

            x = np.linspace(0, 100, shape[0])
            y = np.linspace(0, 100, shape[1])

            vtk_x = vtk.util.numpy_support.numpy_to_vtk(
                num_array=x, deep=True, array_type=data_type
            )
            vtk_y = vtk.util.numpy_support.numpy_to_vtk(
                num_array=y, deep=True, array_type=data_type
            )

            grid.SetXCoordinates(vtk_x)
            grid.SetYCoordinates(vtk_y)
        else:
            raise ValueError("Invalid dimension")

        writer = vtk.vtkRectilinearGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.Write()


# def find_positions_from_contact_angle(contact_angle: float, radius: float):
#     """
#     This is a little helper function that helps find the
#     positions a plane and sphere level-set given some contact
#     angle.
#
#     For now, it assumes that the domain size is 100 by 100 with
#     100 points in either direction.
#     """
#     x_size = 100
#     y_size = 100
#
#     h = 1
#
#     n_points_x = math.ceil(x_size / h)
#     n_points_y = math.ceil(y_size / h)
#
#     x_positions = np.linspace(0, x_size, n_points_x)
#     y_positions = np.linspace(0, y_size, n_points_y)
#
#     # Create the meshgrid
#     x, y = np.meshgrid(x_positions, y_positions)
#
#     # Place the plane level-set somewhere in domain. For now,
#     # we'll always place it at y = 20
#     plane_y_position = 20
#     plane_level_set = PlaneLevelSet([0, 1],
#     [plane_y_position, plane_y_position])
#     plane_level_set_data = plane_level_set.get_value(x, y)
#
#     # Compute the origin of the sphere, given the contact angle
#     sphere_x_position = x_size / 2
#
#     def compute_sphere_y_position(_radius: float):
#         return plane_y_position - _radius * math.cos(contact_angle)
#
#     sphere_y_position = compute_sphere_y_position(radius)
#
#     # Place the sphere level-set
#     sphere_level_set = SphereLevelSet(radius,
#     [sphere_x_position, sphere_y_position])
#     sphere_level_set_data = sphere_level_set.get_value(x, y)
#
#     # Note that this constructs the full sphere, however to simulate
#     # the triple-boundary we have to cut-off the sphere where it intersects
#     # the plane. This is trivial with level-set as we take minimum of the two
#     sets
#     # From there, we can calculate the third phase and recompute the sphere
#     phase
#     interfacial_width = 1
#     combined_level_set = np.minimum(sphere_level_set_data,
#     plane_level_set_data)
#     combined_tanh = level_set_to_tanh(combined_level_set, 1)
#
#     # Convert the level-set fields to tanh
#     sphere_tanh = level_set_to_tanh(sphere_level_set_data, interfacial_width)
#     plane_tanh = level_set_to_tanh(plane_level_set_data, interfacial_width)
#
#     matrix_tanh = 1.0 - combined_tanh
#     sphere_tanh = combined_tanh - plane_tanh
#
#     Output.NumpyToVTK(combined_level_set)
#
#
# find_positions_from_contact_angle(math.pi / 2, 20)
