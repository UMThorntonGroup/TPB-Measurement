import math

import matplotlib.pyplot as plt
import numpy as np

from tpb_measurement import angle, level_set, timer, utilities

if __debug__:
    print("Debug mode is ON")


def find_positions_from_contact_angle(
    contact_angle: float,
    radius: float,
    h: float,
    do_output: bool = False,
    boundary_box_length: int = 1,
):
    """
    This is a little helper function that helps find the
    positions a plane and sphere level-set given some contact
    angle.

    For now, it assumes that the domain size is 100 by 100 with
    100 points in either direction.
    """

    contact_angle = np.pi / 180.0 * contact_angle

    x_size = 100
    y_size = 100

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

    # Create the contact angle object
    contact_angle_object = angle.ContactAngle(
        sphere_level_set_data,
        plane_level_set_data,
        matrix_level_set_data,
        h,
        boundary_box_length,
    )
    contact_angle_object._run_tpb_measurement()

    sphere_normal_data = contact_angle_object.n_1
    plane_normal_data = contact_angle_object.n_2
    matrix_normal_data = contact_angle_object.n_3

    triple_phase_boundary = contact_angle_object.interface_123
    sphere_plane_interface = contact_angle_object.interface_12
    sphere_matrix_interface = contact_angle_object.interface_13
    plane_matrix_interface = contact_angle_object.interface_23

    tpb_direction_candidates = contact_angle_object.tpb_directions

    # Once we've computed the TPB direction, we can compute the contact angles
    # Do to do, we need to find the normal vectors at each of the interfaces
    # project in the direction of the TPB. Importantly, these interface normal
    # vectors must be in a goldilocks spot with respect to the distance from
    # the TPB. To do so, we can implement a quality measure that is a product
    # of the distance and the angle between opposite interface normals. Having
    # the angle approach 180 degrees with a minimal distance will yield the
    # highest quality.
    contact_angle_list = []

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
        for point_set_index, point_set in enumerate(expanded_tpb):
            # Convert the point set to index notation
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
                    sphere_normal_data, local_sphere_plane_indices
                )
            )
            local_normal_sphere_sphere_matrix_interface = (
                vector_data_from_coords(
                    sphere_normal_data, local_sphere_matrix_indices
                )
            )

            local_normal_plane_sphere_plane_interface = (
                vector_data_from_coords(
                    plane_normal_data, local_sphere_plane_indices
                )
            )
            local_normal_plane_plane_matrix_interface = (
                vector_data_from_coords(
                    plane_normal_data, local_plane_matrix_indices
                )
            )

            local_normal_matrix_sphere_matrix_interface = (
                vector_data_from_coords(
                    matrix_normal_data, local_sphere_matrix_indices
                )
            )
            local_normal_matrix_plane_matrix_interface = (
                vector_data_from_coords(
                    matrix_normal_data, local_plane_matrix_indices
                )
            )

            # Check that the sizes are the same between shared interfaces
            utilities.assert_shape(
                local_normal_sphere_sphere_plane_interface,
                local_normal_plane_sphere_plane_interface,
            )
            utilities.assert_shape(
                local_normal_sphere_sphere_matrix_interface,
                local_normal_matrix_sphere_matrix_interface,
            )
            utilities.assert_shape(
                local_normal_plane_plane_matrix_interface,
                local_normal_matrix_plane_matrix_interface,
            )

            # Grab the dot product of the orthogonal interface vectors pairs
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
            utilities.assert_range_with_clipping(local_dot_sphere_plane, -1, 1)
            utilities.assert_range_with_clipping(
                local_dot_sphere_matrix, -1, 1
            )
            utilities.assert_range_with_clipping(local_dot_plane_matrix, -1, 1)

            # Now for the quality measurement. We want to minimize the
            # distance from the TPB and to have the dot product of the
            # vectors approach -1. Importantly, the quality measurement
            # must approach 1 for the ideal case to weight each
            # contribution enough. Approaching 0, would skew results
            # because anything multiplied by zero is itself. Additionally,
            # we have to add some number to the max so we don't multiply
            # by zero for the same reasons as above.
            def renormalize_distance(d):
                return 1 - d / np.max(1.1 * d)

            def renormalize_dot_product(dot):
                return 1 - (dot + 1) / np.max(1.1 * (dot + 1))

            def compute_quality(dot, distance):
                return renormalize_dot_product(
                    dot
                ) + 0.1 * renormalize_distance(distance)

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

            # Grab the max quality normal vectors
            index_sphere_plane = max_quality_index(local_sphere_plane_quality)
            index_sphere_matrix = max_quality_index(
                local_sphere_matrix_quality
            )
            index_plane_matrix = max_quality_index(local_plane_matrix_quality)

            max_quality_normal_sphere_sphere_plane_interface = (
                local_normal_sphere_sphere_plane_interface[index_sphere_plane]
            )
            max_quality_normal_plane_sphere_plane_interface = (
                local_normal_plane_sphere_plane_interface[index_sphere_plane]
            )

            max_quality_normal_sphere_sphere_matrix_interface = (
                local_normal_sphere_sphere_matrix_interface[
                    index_sphere_matrix
                ]
            )
            max_quality_normal_matrix_sphere_matrix_interface = (
                local_normal_matrix_sphere_matrix_interface[
                    index_sphere_matrix
                ]
            )

            max_quality_normal_plane_plane_matrix_interface = (
                local_normal_plane_plane_matrix_interface[index_plane_matrix]
            )
            max_quality_normal_matrix_plane_matrix_interface = (
                local_normal_matrix_plane_matrix_interface[index_plane_matrix]
            )

            # Now that we have the normals, where the quality was good,
            # we can project them onto the most likely TPB direction
            # that was calculated earlier.
            def project_onto_TPB_direction(
                normal, direction, eps: float = 1e-12
            ):
                projection = np.cross(np.cross(direction, normal), direction)
                norm = np.linalg.norm(projection)
                projection /= max(norm, eps)
                return projection

            projected_normal_sphere_sphere_plane_interface = (
                project_onto_TPB_direction(
                    max_quality_normal_sphere_sphere_plane_interface,
                    tpb_direction_candidates[point_set_index],
                )
            )
            projected_normal_plane_sphere_plane_interface = (
                project_onto_TPB_direction(
                    max_quality_normal_plane_sphere_plane_interface,
                    tpb_direction_candidates[point_set_index],
                )
            )

            projected_normal_sphere_sphere_matrix_interface = (
                project_onto_TPB_direction(
                    max_quality_normal_sphere_sphere_matrix_interface,
                    tpb_direction_candidates[point_set_index],
                )
            )
            projected_normal_matrix_sphere_matrix_interface = (
                project_onto_TPB_direction(
                    max_quality_normal_matrix_sphere_matrix_interface,
                    tpb_direction_candidates[point_set_index],
                )
            )

            projected_normal_plane_plane_matrix_interface = (
                project_onto_TPB_direction(
                    max_quality_normal_plane_plane_matrix_interface,
                    tpb_direction_candidates[point_set_index],
                )
            )
            projected_normal_matrix_plane_matrix_interface = (
                project_onto_TPB_direction(
                    max_quality_normal_matrix_plane_matrix_interface,
                    tpb_direction_candidates[point_set_index],
                )
            )

            # Once we have the projected normals, they might not longer satisfy
            # the parallel and opposite clause that we tested in the quality
            # metric above. We correct this, by taking the vector along the
            # direction of the hypotenuse of the other two.
            def correct_opposite_parallel(vec_1, vec_2, eps: float = 1e-12):
                diff = vec_1 - vec_2
                norm = np.linalg.norm(diff)
                diff /= max(norm, eps)

                return diff, -diff

            (
                corrected_normal_sphere_sphere_plane_interface,
                corrected_normal_plane_sphere_plane_interface,
            ) = correct_opposite_parallel(
                projected_normal_sphere_sphere_plane_interface,
                projected_normal_plane_sphere_plane_interface,
            )

            (
                corrected_normal_sphere_sphere_matrix_interface,
                corrected_normal_matrix_sphere_matrix_interface,
            ) = correct_opposite_parallel(
                projected_normal_sphere_sphere_matrix_interface,
                projected_normal_matrix_sphere_matrix_interface,
            )

            (
                corrected_normal_plane_plane_matrix_interface,
                corrected_normal_matrix_plane_matrix_interface,
            ) = correct_opposite_parallel(
                projected_normal_plane_plane_matrix_interface,
                projected_normal_matrix_plane_matrix_interface,
            )

            # Now, we can finally calculate the contact angles of the three
            # phases.
            def compute_contact_angle(vec_1, vec_2):
                dot = np.vecdot(vec_1, vec_2)
                utilities.assert_range_with_clipping(dot, -1, 1)
                angle = 180.0 - 180.0 / np.pi * np.acos(dot)
                return angle

            contact_angle_sphere = compute_contact_angle(
                corrected_normal_sphere_sphere_plane_interface,
                corrected_normal_sphere_sphere_matrix_interface,
            )
            contact_angle_plane = compute_contact_angle(
                corrected_normal_plane_sphere_plane_interface,
                corrected_normal_plane_plane_matrix_interface,
            )
            contact_angle_matrix = compute_contact_angle(
                corrected_normal_matrix_sphere_matrix_interface,
                corrected_normal_matrix_plane_matrix_interface,
            )

            del contact_angle_plane
            del contact_angle_matrix

            contact_angle_list.append(contact_angle_sphere)

    grab_nearby_vectors(boundary_box_length)

    # Print the mean contact angle
    mean_contact_angle = np.mean(np.array(contact_angle_list))
    std_contact_angle = np.std(np.array(contact_angle_list))

    return mean_contact_angle, std_contact_angle


timer_object = timer.Timer()

timer_object.begin("Run")
c_mean, c_std = find_positions_from_contact_angle(
    90, 30, h=0.01, do_output=False, boundary_box_length=5
)
print(c_mean, c_std)
timer_object.end("Run")
timer_object.print_summary()

exit()

h_range = [0.5, 1, 2]
contact_angle_range = np.linspace(40, 140, 10)

plt.figure()
for h in h_range:
    c_mean_list = []
    c_std_list = []

    for c in contact_angle_range:
        c_mean, c_std = find_positions_from_contact_angle(
            c, 30, h=h, do_output=False, boundary_box_length=5
        )

        c_mean_list.append(c_mean)
        c_std_list.append(c_std)

    c_mean_list = np.array(c_mean_list)
    c_std_list = np.array(c_std_list)

    plt.errorbar(
        contact_angle_range,
        np.abs(c_mean_list - contact_angle_range),
        yerr=c_std_list,
        fmt="-o",
        capsize=5,
        label=f"h = {h}",
    )

plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$|\theta_m - \theta_0|$")
plt.legend()
plt.savefig("2d_spherical_cap.png", dpi=300)
