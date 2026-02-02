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

    h = 5

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

    # If we're in 2D, we're going to convert the normal vectors to 3D
    # to make some of the later analysis later.
    if sphere_normal_data.shape[-1] == 2:

        def add_zero_z_component(array):
            zeros = np.zeros((array.shape[0], array.shape[1], 1))
            new_array = np.concatenate((array, zeros), axis=2)
            return new_array

        sphere_normal_data = add_zero_z_component(sphere_normal_data)
        plane_normal_data = add_zero_z_component(plane_normal_data)
        matrix_normal_data = add_zero_z_component(matrix_normal_data)

    # Express the normals for each of the 6 distinct interfaces. Note
    # that the interfaces are given by the 0 level-set between two
    # phases. However, due to the resolution of grid this has to be
    # approximated by half the grid size diagonal ~0.7h
    level_set_tolerance = 0.7 * h
    sphere_zero = np.isclose(
        sphere_level_set_data, 0, atol=level_set_tolerance
    )
    plane_zero = np.isclose(plane_level_set_data, 0, atol=level_set_tolerance)
    matrix_zero = np.isclose(
        matrix_level_set_data, 0, atol=level_set_tolerance
    )

    # Let us consider each triple phase boundary as a single voxel.
    # I don't like this in terms of connectivity, but it should be fine
    triple_phase_boundary = sphere_zero & plane_zero & matrix_zero

    # Throw out TPB voxels that are too close to the boundary
    boundary_box_length = 3
    old_triple_phase_boundary = triple_phase_boundary.copy()
    mask = np.ones_like(triple_phase_boundary, dtype=bool)
    for axis in range(triple_phase_boundary.ndim):
        slicer = [slice(None)] * triple_phase_boundary.ndim

        slicer[axis] = slice(0, boundary_box_length)
        mask[tuple(slicer)] = False

        slicer[axis] = slice(-boundary_box_length, None)
        mask[tuple(slicer)] = False
    triple_phase_boundary &= mask
    if not np.all(old_triple_phase_boundary == triple_phase_boundary):
        print(
            "Warning: some TPB voxels were deleted due to their proximity"
            " to the boundary."
        )
    del old_triple_phase_boundary

    # Also grab the interface masks
    sphere_plane_interface = sphere_zero & plane_zero
    sphere_matrix_interface = sphere_zero & matrix_zero
    plane_matrix_interface = plane_zero & matrix_zero

    # Now we're going to find the triple phase boundary direction
    # this direction is orthogonal to all the normal vectors above.
    def update_a(a, b_1, b_2, b_3):
        def compute_B_ij(i, j):
            return (
                b_1[..., i] * b_1[..., j]
                + b_2[..., i] * b_2[..., j]
                + b_3[..., i] * b_3[..., j]
            )

        B_11 = compute_B_ij(0, 0)
        B_22 = compute_B_ij(1, 1)
        B_33 = compute_B_ij(2, 2)
        B_12 = compute_B_ij(0, 1)
        B_23 = compute_B_ij(1, 2)
        B_13 = compute_B_ij(0, 2)
        a_1 = a[..., 0]
        a_2 = a[..., 1]
        a_3 = a[..., 2]

        def compute_lambda():
            return (
                B_11 * a_1**2
                + B_22 * a_2**2
                + B_33 * a_3**2
                + 2.0
                * (B_12 * a_1 * a_2 + B_23 * a_2 * a_3 + B_13 * a_1 * a_3)
            )

        lambda_values = compute_lambda()

        delta_a_1 = -2.0 * (
            (B_11 - lambda_values) * a_1 + B_12 * a_2 + B_13 * a_3
        )
        delta_a_2 = -2.0 * (
            (B_22 - lambda_values) * a_2 + B_12 * a_1 + B_23 * a_3
        )
        delta_a_3 = -2.0 * (
            (B_33 - lambda_values) * a_3 + B_13 * a_1 + B_23 * a_2
        )

        a[..., 0] += delta_a_1
        a[..., 1] += delta_a_2
        a[..., 2] += delta_a_3

    # Truncate the data to the tpb voxels and save the original data
    sphere_normal_data_original = sphere_normal_data.copy()
    plane_normal_data_original = plane_normal_data.copy()
    matrix_normal_data_original = matrix_normal_data.copy()
    sphere_normal_data = sphere_normal_data[np.where(triple_phase_boundary)]
    plane_normal_data = plane_normal_data[np.where(triple_phase_boundary)]
    matrix_normal_data = matrix_normal_data[np.where(triple_phase_boundary)]

    # Get the initial guess
    tpb_direction_candidates = np.cross(
        matrix_normal_data,
        plane_normal_data,
    )

    iterations = 100
    tolerance = 1.0e-6
    temp_direction = tpb_direction_candidates.copy()
    for i in range(iterations):
        # Update a
        update_a(
            tpb_direction_candidates,
            sphere_normal_data,
            plane_normal_data,
            matrix_normal_data,
        )

        # Compute the relative difference
        rel_difference = np.abs(tpb_direction_candidates - temp_direction)
        print(f"Iteration {i} max difference: {np.max(rel_difference)}")

        # Update the temp
        temp_direction = tpb_direction_candidates

        if np.all(rel_difference < tolerance):
            print(f"Converged at iteration {i}")
            break
        elif i == iterations - 1:
            print(f"Failed to converge after {i} iterations")

    # Once we've computed the TPB direction, we can compute the contact angles
    # Do to do, we need to find the normal vectors at each of the interfaces
    # project in the direction of the TPB. Importantly, these interface normal
    # vectors must be in a goldilocks spot with respect to the distance from
    # the TPB. To do so, we can implement a quality measure that is a product
    # of the distance and the angle between opposite interface normals. Having
    # the angle approach 180 degrees with a minimal distance will yield the
    # highest quality.
    def grab_nearby_vectors(box_length: int):
        # Grab the indices of the TPBs
        tpb = np.array(np.where(triple_phase_boundary))

        # Grab the array of potential offsets
        if triple_phase_boundary.ndim == 2:
            offsets = np.array(
                [
                    (dx, dy)
                    for dx in range(-box_length, box_length + 1)
                    for dy in range(-box_length, box_length + 1)
                ]
            )

        elif triple_phase_boundary.ndim == 3:
            offsets = np.array(
                [
                    (dx, dy, dz)
                    for dx in range(-box_length, box_length + 1)
                    for dy in range(-box_length, box_length + 1)
                    for dz in range(-box_length, box_length + 1)
                ]
            )
        else:
            raise ValueError(
                "Triple phase boundary ndim not supported:"
                f"{triple_phase_boundary.ndim}"
            )

        # Broadcast the addition of the offsets
        # Note: tpb has mismatched indices. Hopefully, this doesn't
        # mess up some of the stuff above...
        expanded_tpb = np.transpose(tpb)[:, None, :] + offsets[None, :, :]

        # Grab the distance from the TPB, which is simply the norm of the
        # offset vector
        tpb_distance = np.linalg.norm(offsets, axis=-1)

        # Take the broadcasted region and filter based on the three interface
        # types
        for point_set in expanded_tpb:
            # Convert the point set to index notation
            # Why is python and numpy like this... indexing ..............
            # ............................................................
            # ahhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
            #
            point_index_set = tuple(np.transpose(point_set))

            # Grab the local mask from the neighbors
            local_sphere_plane = sphere_plane_interface[point_index_set]
            local_sphere_matrix = sphere_matrix_interface[point_index_set]
            local_plane_matrix = plane_matrix_interface[point_index_set]

            # Print a warning if we don't find enough points
            if np.sum(local_sphere_plane) == 0:
                print(
                    "Warning: box scanning size too small, found 0 "
                    "points at the sphere-plane interface."
                )
            if np.sum(local_sphere_matrix) == 0:
                print(
                    "Warning: box scanning size too small, found 0 "
                    "points at the sphere-matrix interface."
                )
            if np.sum(local_plane_matrix) == 0:
                print(
                    "Warning: box scanning size too small, found 0 "
                    "points at the plane-matrix interface."
                )

            def vector_data_from_coords(data, coords):
                coords = np.asarray(coords)
                index_tuple = tuple(
                    coords[:, j] for j in range(coords.shape[1])
                )
                return data[index_tuple]

            # Grab the nearby normals
            local_sphere_plane_indices = point_set[
                np.where(local_sphere_plane)
            ]
            local_sphere_matrix_indices = point_set[
                np.where(local_sphere_matrix)
            ]
            local_plane_matrix_indices = point_set[
                np.where(local_plane_matrix)
            ]

            local_normal_sphere_sphere_plane_interface = (
                vector_data_from_coords(
                    sphere_normal_data_original, local_sphere_plane_indices
                )
            )
            local_normal_sphere_sphere_matrix_interface = (
                vector_data_from_coords(
                    sphere_normal_data_original, local_sphere_matrix_indices
                )
            )

            local_normal_plane_sphere_plane_interface = (
                vector_data_from_coords(
                    plane_normal_data_original, local_sphere_plane_indices
                )
            )
            local_normal_plane_plane_matrix_interface = (
                vector_data_from_coords(
                    plane_normal_data_original, local_plane_matrix_indices
                )
            )

            local_normal_matrix_sphere_matrix_interface = (
                vector_data_from_coords(
                    matrix_normal_data_original, local_sphere_matrix_indices
                )
            )
            local_normal_matrix_plane_matrix_interface = (
                vector_data_from_coords(
                    matrix_normal_data_original, local_plane_matrix_indices
                )
            )

            # Check that the sizes are the same between shared interfaces
            if (
                local_normal_sphere_sphere_plane_interface.shape
                != local_normal_plane_sphere_plane_interface.shape
            ):
                raise ValueError("Shape mismatch at sphere-plane interface")
            if (
                local_normal_sphere_sphere_matrix_interface.shape
                != local_normal_matrix_sphere_matrix_interface.shape
            ):
                raise ValueError("Shape mismatch at sphere-matrix interface")
            if (
                local_normal_plane_plane_matrix_interface.shape
                != local_normal_matrix_plane_matrix_interface.shape
            ):
                raise ValueError("Shape mismatch at plane-matrix interface")

            # Grab the dot product of the orthogonal interface vectors pairs
            def check_valid_range(
                data,
                min: float = -1.0,
                max: float = 1.0,
                clip_tolerance: float = None,
            ):
                if clip_tolerance is not None:
                    mask = (data < (min - clip_tolerance)) | (
                        data > (max + clip_tolerance)
                    )
                    data[~mask] = np.clip(data[~mask], min, max)
                else:
                    mask = (data < min) | (data > max)
                if np.any(mask):
                    raise ValueError(
                        f"Invalid range for data: {data[np.where(mask)]}"
                    )

            local_dot_sphere_plane = np.vecdot(
                local_normal_sphere_sphere_plane_interface,
                local_normal_plane_sphere_plane_interface,
            )
            local_dot_sphere_matrix = np.vecdot(
                local_normal_sphere_sphere_matrix_interface,
                local_normal_matrix_sphere_matrix_interface,
            )
            local_dot_plane_matrix = np.vecdot(
                local_normal_plane_plane_matrix_interface,
                local_normal_matrix_plane_matrix_interface,
            )
            check_valid_range(local_dot_sphere_plane, clip_tolerance=1e-6)
            check_valid_range(local_dot_sphere_matrix, clip_tolerance=1e-6)
            check_valid_range(local_dot_plane_matrix, clip_tolerance=1e-6)

            # Now for the quality measurement. We want to minimize the
            # distance from the TPB and to have the dot product of the
            # vectors approach -1. Importantly, the quality measurement
            # must approach 1 for the ideal case to weight each
            # contribution enough. Approaching 0, would skew results
            # because anything multiplied by zero is itself. Additionally
            # we have to add some number to the max so we don't multiply
            # by zero for the same reasosns as above.
            def renormalize_distance(d):
                return 1 - d / np.max(1.1 * d)

            def renormalize_dot_product(d):
                return 1 - (d + 1) / np.max(1.1 * (d + 1))

            def compute_quality(dot, distance):
                return renormalize_dot_product(dot) * renormalize_distance(
                    distance
                )

            local_sphere_plane_quality = compute_quality(
                local_dot_sphere_plane,
                tpb_distance[np.where(local_sphere_plane)],
            )
            local_sphere_matrix_quality = compute_quality(
                local_dot_sphere_matrix,
                tpb_distance[np.where(local_sphere_matrix)],
            )
            local_plane_matrix_quality = compute_quality(
                local_dot_plane_matrix,
                tpb_distance[np.where(local_plane_matrix)],
            )

            def max_quality_index(quality):
                return np.argmax(quality)

            def print_info(context, dot, distance, quality):
                print(
                    f"{context}:\n"
                    f"  dot: {dot[max_quality_index(quality)]}\n"
                    f"  distance: {distance[max_quality_index(quality)]}\n"
                    f"  quality: {quality[max_quality_index(quality)]}\n"
                )

            print_info(
                "sphere-plane",
                local_dot_sphere_plane,
                tpb_distance[np.where(local_sphere_plane)],
                local_sphere_plane_quality,
            )
            print_info(
                "sphere-matrix",
                local_dot_sphere_matrix,
                tpb_distance[np.where(local_sphere_matrix)],
                local_sphere_matrix_quality,
            )
            print_info(
                "plane-matrix",
                local_dot_plane_matrix,
                tpb_distance[np.where(local_plane_matrix)],
                local_plane_matrix_quality,
            )

    grab_nearby_vectors(boundary_box_length)

    # We're going to switch up the notation a little such that the
    # sphere is the Ni phase, the plane is the YSZ phase, and the matrix
    # is the pore phase
    # normal_ni_pore_interface = sphere_normal_data[
    #     np.where(sphere_matrix_interface & triple_phase_neighborhood)
    # ]
    # normal_ni_ysz_interface = sphere_normal_data[
    #     np.where(sphere_plane_interface & triple_phase_neighborhood)
    # ]
    # normal_ysz_ni_interface = plane_normal_data[
    #     np.where(sphere_plane_interface & triple_phase_neighborhood)
    # ]
    # normal_ysz_pore_interface = plane_normal_data[
    #     np.where(plane_matrix_interface & triple_phase_neighborhood)
    # ]
    # normal_pore_ni_interface = matrix_normal_data[
    #     np.where(sphere_matrix_interface & triple_phase_neighborhood)
    # ]
    # normal_pore_ysz_interface = matrix_normal_data[
    #     np.where(plane_matrix_interface & triple_phase_neighborhood)
    # ]

    # Grab the angles from each of the opposite pairs
    # Note that we have unequal arrays of the normal pairs so we need to do
    # some matrix multiplication to get all enumerations
    # dot = np.ravel(
    #     np.matmul(
    #         normal_ni_pore_interface, np.transpose(normal_ni_ysz_interface)
    #     )
    # )
    # dot = np.clip(dot, -1, 1)
    # angle_ni = 180.0 - 180.0 / np.pi * np.acos(dot)
    # dot = np.ravel(
    #     np.matmul(
    #         normal_ysz_ni_interface, np.transpose(normal_ysz_pore_interface)
    #     )
    # )
    # dot = np.clip(dot, -1, 1)
    # angle_ysz = 180.0 - 180.0 / np.pi * np.acos(dot)
    # dot = np.ravel(
    #     np.matmul(
    #         normal_pore_ni_interface, np.transpose(normal_pore_ysz_interface)
    #     )
    # )
    # dot = np.clip(dot, -1, 1)
    # angle_pore = 180.0 - 180.0 / np.pi * np.acos(dot)

    # Now that we've got a bunch of potential angles we need to downselect
    # angle_ni = np.unique(angle_ni)
    # angle_ysz = np.unique(angle_ysz)
    # angle_pore = np.unique(angle_pore)
    # print(angle_ni)
    # print(angle_ysz)
    # print(angle_pore)

    # Output the data
    fields = {
        "sphere": sphere_level_set_data,
        "plane": plane_level_set_data,
        "matrix": matrix_level_set_data,
        "sphere_plane": sphere_plane_interface,
        "sphere_matrix": sphere_matrix_interface,
        "plane_matrix": plane_matrix_interface,
        "triple_phase_boundary": triple_phase_boundary,
        "sphere_normal": sphere_normal_data_original,
        "plane_normal": plane_normal_data_original,
        "matrix_normal": matrix_normal_data_original,
    }
    output.Output.numpy_to_rectilinear_vtk(
        fields, domain_size=np.array([x_size, y_size])
    )


find_positions_from_contact_angle(math.pi / 2, 30)
