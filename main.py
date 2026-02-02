import math

import numpy as np
import matplotlib.pyplot as plt
from tpb_measurement import level_set, numerics, output, angle


def find_positions_from_contact_angle(
    contact_angle: float, radius: float, h: float, do_output: bool = False
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
    matrix_level_set_data = -np.minimum(sphere_level_set_data, plane_level_set_data)
    sphere_level_set_data = np.maximum(sphere_level_set_data, -plane_level_set_data)

    # Reshape the data
    sphere_level_set_data = np.reshape(sphere_level_set_data, original_shape)
    plane_level_set_data = np.reshape(plane_level_set_data, original_shape)
    matrix_level_set_data = np.reshape(matrix_level_set_data, original_shape)

    # Create the the contact angle object
    boundary_box_length = 5
    contact_angle_object = angle.ContactAngle(
        sphere_level_set_data,
        plane_level_set_data,
        matrix_level_set_data,
        h,
        boundary_box_length,
    )
    contact_angle_object._get_normals()
    contact_angle_object._get_masks()
    contact_angle_object._find_triple_boundary_direction()

    sphere_normal_data = contact_angle_object.n_1
    plane_normal_data = contact_angle_object.n_2
    matrix_normal_data = contact_angle_object.n_3

    triple_phase_boundary = contact_angle_object.interface_123
    sphere_plane_interface = contact_angle_object.interface_12
    sphere_matrix_interface = contact_angle_object.interface_13
    plane_matrix_interface = contact_angle_object.interface_23

    tpb_direction_candidates = contact_angle_object.tpb_directions

    # Truncate the data to the tpb voxels and save the original data
    sphere_normal_data_original = sphere_normal_data.copy()
    plane_normal_data_original = plane_normal_data.copy()
    matrix_normal_data_original = matrix_normal_data.copy()

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
                index_tuple = tuple(coords[:, j] for j in range(coords.shape[1]))
                return data[index_tuple]

            # Grab the nearby normals
            local_sphere_plane_indices = point_set[np.where(local_sphere_plane)]
            local_sphere_matrix_indices = point_set[np.where(local_sphere_matrix)]
            local_plane_matrix_indices = point_set[np.where(local_plane_matrix)]

            local_normal_sphere_sphere_plane_interface = vector_data_from_coords(
                sphere_normal_data_original, local_sphere_plane_indices
            )
            local_normal_sphere_sphere_matrix_interface = vector_data_from_coords(
                sphere_normal_data_original, local_sphere_matrix_indices
            )

            local_normal_plane_sphere_plane_interface = vector_data_from_coords(
                plane_normal_data_original, local_sphere_plane_indices
            )
            local_normal_plane_plane_matrix_interface = vector_data_from_coords(
                plane_normal_data_original, local_plane_matrix_indices
            )

            local_normal_matrix_sphere_matrix_interface = vector_data_from_coords(
                matrix_normal_data_original, local_sphere_matrix_indices
            )
            local_normal_matrix_plane_matrix_interface = vector_data_from_coords(
                matrix_normal_data_original, local_plane_matrix_indices
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
                    raise ValueError(f"Invalid range for data: {data[np.where(mask)]}")

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
                return (
                    renormalize_dot_product(dot) * 0.5 * renormalize_distance(distance)
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

            # Grab the max quality normal vectors
            index_sphere_plane = max_quality_index(local_sphere_plane_quality)
            index_sphere_matrix = max_quality_index(local_sphere_matrix_quality)
            index_plane_matrix = max_quality_index(local_plane_matrix_quality)

            max_quality_normal_sphere_sphere_plane_interface = (
                local_normal_sphere_sphere_plane_interface[index_sphere_plane]
            )
            max_quality_normal_plane_sphere_plane_interface = (
                local_normal_plane_sphere_plane_interface[index_sphere_plane]
            )

            max_quality_normal_sphere_sphere_matrix_interface = (
                local_normal_sphere_sphere_matrix_interface[index_sphere_matrix]
            )
            max_quality_normal_matrix_sphere_matrix_interface = (
                local_normal_matrix_sphere_matrix_interface[index_sphere_matrix]
            )

            max_quality_normal_plane_plane_matrix_interface = (
                local_normal_plane_plane_matrix_interface[index_plane_matrix]
            )
            max_quality_normal_matrix_plane_matrix_interface = (
                local_normal_matrix_plane_matrix_interface[index_plane_matrix]
            )

            # Print some info about the max quality
            print(
                "\nSphere-Plane\n"
                f"  dot product: {local_dot_sphere_plane[index_sphere_plane]}\n"
                f"  distance: {tpb_distance[np.where(local_sphere_plane)][index_sphere_plane]}\n"
            )
            print(
                "\nSphere-Matrix\n"
                f"  dot product: {local_dot_sphere_matrix[index_sphere_matrix]}\n"
                f"  distance: {tpb_distance[np.where(local_sphere_matrix)][index_sphere_matrix]}\n"
            )
            print(
                "\nPlane-Matrix\n"
                f"  dot product: {local_dot_plane_matrix[index_plane_matrix]}\n"
                f"  distance: {tpb_distance[np.where(local_plane_matrix)][index_plane_matrix]}\n"
            )

            # Now that we have the normals, where the quality was good,
            # we can project them onto the most likely TPB direction
            # that was calculated earlier.
            def project_onto_TPB_direction(normal, direction):
                projection = np.cross(np.cross(direction, normal), direction)
                projection /= np.linalg.norm(projection)

                return projection

            projected_normal_sphere_sphere_plane_interface = project_onto_TPB_direction(
                max_quality_normal_sphere_sphere_plane_interface,
                tpb_direction_candidates[point_set_index],
            )
            projected_normal_plane_sphere_plane_interface = project_onto_TPB_direction(
                max_quality_normal_plane_sphere_plane_interface,
                tpb_direction_candidates[point_set_index],
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

            projected_normal_plane_plane_matrix_interface = project_onto_TPB_direction(
                max_quality_normal_plane_plane_matrix_interface,
                tpb_direction_candidates[point_set_index],
            )
            projected_normal_matrix_plane_matrix_interface = project_onto_TPB_direction(
                max_quality_normal_matrix_plane_matrix_interface,
                tpb_direction_candidates[point_set_index],
            )

            # Once we have the projected normals, they might not longer satisfy
            # the parallel and opposite clause that we tested in the quality
            # metric above. We correct this, by taking the vector along the
            # direction of the hypotenuse of the other two.
            def correct_opposite_parallel(vec_1, vec_2):
                diff = vec_1 - vec_2
                diff /= np.linalg.norm(diff)

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

            print(f"Sphere contact angle {contact_angle_sphere}")
            print(f"Plane contact angle {contact_angle_plane}")
            print(f"Matrix contact angle {contact_angle_matrix}")
            print(
                f"Total angle {contact_angle_sphere + contact_angle_plane + contact_angle_matrix}\n"
            )

            contact_angle_list.append(contact_angle_sphere)

    grab_nearby_vectors(boundary_box_length)

    # Print the mean contact angle
    mean_contact_angle = np.mean(np.array(contact_angle_list))

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
    if do_output:
        output.Output.numpy_to_rectilinear_vtk(
            fields, domain_size=np.array([x_size, y_size])
        )

    return mean_contact_angle


find_positions_from_contact_angle(40, 30, h=0.05, do_output=True)
exit()

h_range = np.logspace(-1.5, 0, 4)
contact_angle_range = np.linspace(40, 140, 20)
contact_angle_error = np.zeros((len(h_range), len(contact_angle_range)))
for i, h in enumerate(h_range):
    for j, ideal_contact_angle in enumerate(contact_angle_range):
        c = find_positions_from_contact_angle(ideal_contact_angle, 30, h)
        contact_angle_error[i, j] = np.abs(c - ideal_contact_angle)

H, CA = np.meshgrid(contact_angle_range, h_range)
plt.figure(figsize=(7, 5))

print(contact_angle_error)

plt.imshow(
    contact_angle_error,
    origin="lower",
    aspect="auto",
    interpolation="none",
    extent=[
        contact_angle_range.min(),
        contact_angle_range.max(),
        h_range.min(),
        h_range.max(),
    ],
)

plt.colorbar(label="Computed contact angle")
plt.xlabel("Ideal contact angle (deg)")
plt.ylabel("h")
plt.yscale("log")


plt.tight_layout()
plt.savefig("test.png", dpi=300)
