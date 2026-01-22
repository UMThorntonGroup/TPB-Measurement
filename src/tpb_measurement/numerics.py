import numpy as np


class NormalData:
    def __init__(
        self,
        data: np.ndarray,
        ghost_layer_depth: int = 1,
        pad_mode: str = "reflect",
    ):
        self.data = np.transpose(data)
        self.data_shape = np.shape(self.data)

        self.ndim = self.data.ndim
        self.pad_depth = ghost_layer_depth
        self.pad_mode = pad_mode
        self.slices = tuple(
            slice(self.pad_depth, -self.pad_depth) for _ in range(self.ndim)
        )

        self.ghost_data = np.pad(self.data, self.pad_depth, mode=self.pad_mode)
        self.ghost_data_shape = np.shape(self.ghost_data)

        self.gradient_scratch_data = np.zeros(self.data_shape + (self.ndim,))

    def get_normal(self, h: float = 1):
        # Compute the velocity for the upwinding as the central different
        # gradient note that this velocity must all be unity
        self._central_difference_gradient(h)
        self.gradient_scratch_data[
            np.where(self.gradient_scratch_data > 0)
        ] = 1
        self.gradient_scratch_data[
            np.where(self.gradient_scratch_data < 0)
        ] = -1
        # First order upwind to get the gradients
        self._first_order_upwind(self.gradient_scratch_data, h)

        # Finally, find the normal
        self._find_normal()

        return self.gradient_scratch_data

    def _find_normal(self, epsilon: float = 1e-6):
        # Compute vector magnitude (L2 norm)
        norm = np.linalg.norm(
            self.gradient_scratch_data, axis=-1, keepdims=True
        )

        # Avoid division by zero
        norm = np.maximum(norm, epsilon)

        # Normalize
        self.gradient_scratch_data /= norm

    def _remove_ghosts(self):
        return self.ghost_data[self.slices]

    def _update_ghosts(self):
        self.ghost_data = np.pad(self.data, self.pad_depth, mode=self.pad_mode)

    def _central_difference_gradient(self, h: float = 1):
        for axis in range(self.ndim):
            grad = (
                np.roll(self.ghost_data, -1, axis=axis)
                - np.roll(self.ghost_data, 1, axis=axis)
            ) / (2.0 * h)
            # Truncate the data and add to the scratch
            self.gradient_scratch_data[..., axis] = grad[self.slices]

    def _first_order_upwind(self, velocity: np.ndarray, h: float = 1):
        for axis in range(self.ndim):
            # Take the forward gradient
            fwd_grad = (
                np.roll(self.ghost_data, -1, axis=axis) - self.ghost_data
            ) / h
            # Take the backward gradient
            bwd_grad = (
                self.ghost_data - np.roll(self.ghost_data, 1, axis=axis)
            ) / h

            # Find the mask for upwinding
            fwd_vel = np.minimum(velocity[..., axis], 0)
            bwd_vel = np.maximum(velocity[..., axis], 0)

            # Find the upwind gradient
            grad = (
                fwd_grad[self.slices] * fwd_vel
                + bwd_grad[self.slices] * bwd_vel
            )

            # Add back into the scratch data
            self.gradient_scratch_data[..., axis] = grad
