import math

import numpy as np

from tpb_measurement import level_set, numerics, output


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
    original_shape = x.shape
    # Flatten the data
    x = x.flatten()
    y = y.flatten()

    # Place the plane level-set somewhere in domain. For now,
    # we'll always place it at y = 20
    plane_y_position = 20
    plane_level_set = level_set.PlaneLevelSet(
        [0, 1], [plane_y_position, plane_y_position]
    )
    plane_level_set_data = plane_level_set.get_value(x, y)

    # Compute the origin of the sphere, given the contact angle
    sphere_x_position = x_size / 2

    def compute_sphere_y_position(_radius: float):
        return plane_y_position - _radius * math.cos(contact_angle)

    sphere_y_position = compute_sphere_y_position(radius)

    # Place the sphere level-set
    sphere_level_set = level_set.SphereLevelSet(
        radius, [sphere_x_position, sphere_y_position]
    )
    sphere_level_set_data = sphere_level_set.get_value(x, y)

    # Note that this constructs the full sphere, however to simulate
    # the triple-boundary we have to cut-off the sphere where it intersects
    # the plane. This is trivial with level-set as we take minimum of the two
    # sets. From there, we can calculate the third phase and recompute the
    # sphere phase
    matrix_level_set_data = -np.minimum(
        sphere_level_set_data, plane_level_set_data
    )
    sphere_level_set_data = np.maximum(
        sphere_level_set_data, -plane_level_set_data
    )

    # Reshape the data
    sphere_level_set_data = np.reshape(sphere_level_set_data, original_shape)
    plane_level_set_data = np.reshape(plane_level_set_data, original_shape)
    matrix_level_set_data = np.reshape(matrix_level_set_data, original_shape)

    # Compute the normals for each of the data sets
    sphere_normal = numerics.NormalData(sphere_level_set_data)
    sphere_normal_data = sphere_normal.get_normal(h)
    plane_normal = numerics.NormalData(plane_level_set_data)
    plane_normal_data = plane_normal.get_normal(h)
    matrix_normal = numerics.NormalData(matrix_level_set_data)
    matrix_normal_data = matrix_normal.get_normal(h)

    # Express the normals for each of the 6 distinct interfaces. Note
    # that the interfaces are given by the 0 level-set between two
    # phases. However, due to the resolution of grid this has to be
    # approximated by half the grid size diagonal ~0.7h
    tolerance = 0.7 * h
    sphere_zero = np.isclose(sphere_level_set_data, 0, atol=tolerance)
    plane_zero = np.isclose(plane_level_set_data, 0, atol=tolerance)
    matrix_zero = np.isclose(matrix_level_set_data, 0, atol=tolerance)

    triple_phase_boundary = sphere_zero & plane_zero & matrix_zero

    sphere_plane_interface = sphere_zero & plane_zero
    sphere_matrix_interface = sphere_zero & matrix_zero
    plane_matrix_interface = plane_zero & matrix_zero

    # We also create a looser tolerance for the TPB neighborhood
    tolerance = 2 * h
    sphere_zero = np.isclose(sphere_level_set_data, 0, atol=tolerance)
    plane_zero = np.isclose(plane_level_set_data, 0, atol=tolerance)
    matrix_zero = np.isclose(matrix_level_set_data, 0, atol=tolerance)
    triple_phase_neighborhood = sphere_zero & plane_zero & matrix_zero

    # We're going to switch up the notation a little such that the
    # sphere is the Ni phase, the plane is the YSZ phase, and the matrix
    # is the pore phase
    normal_ni_pore_interface = sphere_normal_data[
        np.where(sphere_matrix_interface & triple_phase_neighborhood)
    ]
    normal_ni_ysz_interface = sphere_normal_data[
        np.where(sphere_plane_interface & triple_phase_neighborhood)
    ]
    normal_ysz_ni_interface = plane_normal_data[
        np.where(sphere_plane_interface & triple_phase_neighborhood)
    ]
    normal_ysz_pore_interface = plane_normal_data[
        np.where(plane_matrix_interface & triple_phase_neighborhood)
    ]
    normal_pore_ni_interface = matrix_normal_data[
        np.where(sphere_matrix_interface & triple_phase_neighborhood)
    ]
    normal_pore_ysz_interface = matrix_normal_data[
        np.where(plane_matrix_interface & triple_phase_neighborhood)
    ]

    # Grab the angles from each of the opposite pairs
    # Note that we have unequal arrays of the normal pairs so we need to do
    # some matrix multiplication to get all enumerations
    dot = np.ravel(
        np.matmul(
            normal_ni_pore_interface, np.transpose(normal_ni_ysz_interface)
        )
    )
    angle_ni = 180.0 - 180.0 / np.pi * np.acos(dot)
    dot = np.ravel(
        np.matmul(
            normal_ysz_ni_interface, np.transpose(normal_ysz_pore_interface)
        )
    )
    angle_ysz = 180.0 - 180.0 / np.pi * np.acos(dot)
    dot = np.ravel(
        np.matmul(
            normal_pore_ni_interface, np.transpose(normal_pore_ysz_interface)
        )
    )
    angle_pore = 180.0 - 180.0 / np.pi * np.acos(dot)
    print(angle_ni)
    print(angle_ysz)
    print(angle_pore)

    # Output the data
    fields = {
        "sphere": sphere_level_set_data,
        "plane": plane_level_set_data,
        "matrix": matrix_level_set_data,
        "sphere_plane": sphere_plane_interface,
        "sphere_matrix": sphere_matrix_interface,
        "plane_matrix": plane_matrix_interface,
        "triple_phase_boundary": triple_phase_boundary,
        "triple_phase_neighborhood": triple_phase_neighborhood,
    }
    output.Output.numpy_to_rectilinear_vtk(
        fields, domain_size=np.array([x_size, y_size])
    )


find_positions_from_contact_angle(math.pi / 2, 20)
