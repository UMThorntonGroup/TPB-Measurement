import numpy as np

from tpb_measurement import numerics


class ContactAngle:
    def __init__(self, level_set_1, level_set_2, level_set_3, h, b_length):
        # Level-set fields for each of the three phases
        # TODO: Add check here to make sure they sum correctly
        self.ls_1 = level_set_1
        self.ls_2 = level_set_2
        self.ls_3 = level_set_3

        # Grid spacing (uniform)
        self.h = h

        # Boundary box spacing
        self.b_length = b_length

        # Max number of iterations and tolerance for the search of the most
        # likely triple boundary direction
        self.iterations = 100
        self.tolerance = 1e-6

        # Normal vectors of the three fields. These are computed with a
        # central difference and will always have three components.
        self.n_1 = None
        self.n_2 = None
        self.n_3 = None

        # Interface masks for all combinations
        self.interface_12 = None
        self.interface_13 = None
        self.interface_23 = None
        self.interface_123 = None

        # Triple boundary directions
        self.tpb_directions = None

    def _get_normals(self):
        normal_object_1 = numerics.NormalData(self.ls_1)
        self.n_1 = normal_object_1.get_normal(self.h).copy()
        del normal_object_1
        normal_object_2 = numerics.NormalData(self.ls_2)
        self.n_2 = normal_object_2.get_normal(self.h).copy()
        del normal_object_2
        normal_object_3 = numerics.NormalData(self.ls_3)
        self.n_3 = normal_object_3.get_normal(self.h).copy()
        del normal_object_3

        # If we're in 2D, we're going to convert the normal vectors to 3D
        # to make some of the later analysis later.
        if self.n_1.shape[-1] == 2:

            def add_zero_z_component(array):
                zeros = np.zeros((array.shape[0], array.shape[1], 1))
                new_array = np.concatenate((array, zeros), axis=2)
                return new_array

            self.n_1 = add_zero_z_component(self.n_1)
            self.n_2 = add_zero_z_component(self.n_2)
            self.n_3 = add_zero_z_component(self.n_3)

    def _get_masks(self):
        # Note that the interfaces are given by the 0 level-set between two
        # or more phases. As such, we must take into account the grid
        # resolution. For this reason, I use 0.7h as some tolerance.
        # TODO: Pick a better tolerance this isn't just the 2D diagonal
        # length
        grid_tol = 0.7 * self.h

        mask_1 = np.isclose(self.ls_1, 0, atol=grid_tol)
        mask_2 = np.isclose(self.ls_2, 0, atol=grid_tol)
        mask_3 = np.isclose(self.ls_3, 0, atol=grid_tol)

        # Grab the binary interfaces
        self.interface_12 = mask_1 & mask_2
        self.interface_13 = mask_1 & mask_3
        self.interface_23 = mask_2 & mask_3

        # Let us consider each triple phase boundary as a single
        # voxel. I don't like this in terms of connectivity,
        # but it should be fine
        self.interface_123 = mask_1 & mask_2 & mask_3

        # Throw out TPB voxels that are too close to the boundary
        old_interface_123 = self.interface_123.copy()
        mask = np.ones_like(old_interface_123, dtype=bool)
        for axis in range(old_interface_123.ndim):
            slicer = [slice(None)] * old_interface_123.ndim

            slicer[axis] = slice(0, self.b_length)
            mask[tuple(slicer)] = False

            slicer[axis] = slice(-self.b_length, None)
            mask[tuple(slicer)] = False
        self.interface_123 &= mask
        if not np.all(old_interface_123 == self.interface_123):
            print(
                "Warning: some TPB voxels were deleted due to their proximity"
                " to the boundary."
            )
        del old_interface_123

        if np.sum(self.interface_123) == 0:
            raise ValueError("No triple phase boundaries have been found")

    def _find_triple_boundary_direction(self):
        # Now find the direction of the triple phase boundary. This
        # direction is orthogonal to all the normal vectors above,
        # so we can iterate on this minimization project with the
        # function below.
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

        # Downselect the normals vectors to those only at the TPB voxels
        normal_1 = self.n_1[np.where(self.interface_123)]
        normal_2 = self.n_2[np.where(self.interface_123)]
        normal_3 = self.n_3[np.where(self.interface_123)]

        # Get the initial guess
        self.tpb_directions = np.cross(normal_2, normal_3)

        # Copy the vector so we have one to update
        temp_direction = self.tpb_directions.copy()

        for i in range(self.iterations):
            update_a(self.tpb_directions, normal_1, normal_2, normal_3)

            # Compute the relative difference
            rel_difference = np.abs(self.tpb_directions - temp_direction)
            print(f"Iteration {i} max difference: {np.max(rel_difference)}")

            # Update the temp
            temp_direction = self.tpb_directions

            if np.all(rel_difference < self.tolerance):
                print(f"Converged at iteration {i}")
                break
            elif i == self.iterations - 1:
                print(f"Failed to converge after {i} iterations")
