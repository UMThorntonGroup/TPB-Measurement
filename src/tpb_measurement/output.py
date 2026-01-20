from pathlib import Path

import numpy as np
import vtk.util.numpy_support


class Output:
    def __init__(self):
        pass

    @staticmethod
    def _coord_setup(
        shape: np.ndarray,
        domain_size: np.ndarray,
        grid: vtk.vtkRectilinearGrid,
        data_type=vtk.VTK_FLOAT,
    ):
        if len(shape) == 3:
            grid.SetDimensions(shape[0], shape[1], shape[2])

            if domain_size is None:
                x = np.linspace(0, 1, shape[0])
                y = np.linspace(0, 1, shape[1])
                z = np.linspace(0, 1, shape[2])
            else:
                x = np.linspace(0, domain_size[0], shape[0])
                y = np.linspace(0, domain_size[1], shape[1])
                z = np.linspace(0, domain_size[2], shape[2])

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

            if domain_size is None:
                x = np.linspace(0, 1, shape[0])
                y = np.linspace(0, 1, shape[1])
            else:
                x = np.linspace(0, domain_size[0], shape[0])
                y = np.linspace(0, domain_size[1], shape[1])

            vtk_x = vtk.util.numpy_support.numpy_to_vtk(
                num_array=x, deep=True, array_type=data_type
            )
            vtk_y = vtk.util.numpy_support.numpy_to_vtk(
                num_array=y, deep=True, array_type=data_type
            )

            grid.SetXCoordinates(vtk_x)
            grid.SetYCoordinates(vtk_y)
        elif len(shape) == 1:
            grid.SetDimensions(shape[0], 1, 1)

            if domain_size is None:
                x = np.linspace(0, 1, shape[0])
            else:
                x = np.linspace(0, domain_size[0], shape[0])

            vtk_x = vtk.util.numpy_support.numpy_to_vtk(
                num_array=x, deep=True, array_type=data_type
            )

            grid.SetXCoordinates(vtk_x)
        else:
            raise ValueError("Invalid dimension")

    @staticmethod
    def _writer_setup(filename: str, grid: vtk.vtkRectilinearGrid):
        # Grab the extension of the filename
        ext = Path(filename).suffix.lower()

        if ext == ".vtk":
            writer = vtk.vtkRectilinearGridWriter()
            writer.SetFileName(filename)
            writer.SetInputData(grid)
            writer.Write()

        elif ext == ".vtr":
            writer = vtk.vtkXMLRectilinearGridWriter()
            writer.SetFileName(filename)
            writer.SetInputData(grid)
            writer.SetDataModeToBinary()
            writer.Write()

        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                "Use '.vtk' (legacy) or '.vtr' (VTK XML)."
            )

    @staticmethod
    def numpy_to_rectilinear_vtk(
        fields: dict[str, np.ndarray],
        filename: str = "test.vtr",
        time: float = 0.0,
        domain_size: np.ndarray = None,
    ):
        data_type = vtk.VTK_FLOAT

        # Use first field to define grid. To do so, we next to determine
        # if it's a scalar or vector
        first_key = next(iter(fields))
        first_data = fields[first_key]
        if first_data.ndim == 2:
            shape = first_data.shape
        elif first_data.ndim == 3 and first_data.shape[-1] in (2, 3):
            # NOTE: This can technically misinterpret data is our domain
            # is 2 or 3 points in a certain direction
            shape = first_data.shape[:-1]
        elif first_data.ndim == 3:
            shape = first_data.shape
        elif first_data.ndim == 4 and first_data.shape[-1] in (2, 3):
            shape = first_data.shape[:-1]
        else:
            raise ValueError("Field must be scalar or vector.")

        # Create the rectilinear grid object
        grid = vtk.vtkRectilinearGrid()

        # Add the data
        for name, data in fields.items():
            # 2D scalar
            if data.ndim == 2:
                if data.shape != shape:
                    raise ValueError(f"Field '{name}' has mismatched shape")
                flat_data = data.flatten()
                comps = 1
            # 2D vector
            elif data.ndim == 3 and data.shape[-1] in (2, 3):
                if data.shape[:-1] != shape:
                    raise ValueError(f"Field '{name}' has mismatched shape")
                flat_data = data.reshape(-1, data.shape[-1])
                comps = data.shape[-1]
            # 3D scalar
            elif data.ndim == 3:
                if data.shape != shape:
                    raise ValueError(f"Field '{name}' has mismatched shape")
                flat_data = data.flatten()
                comps = 1
            # 3D vector
            elif data.ndim == 4 and data.shape[-1] in (2, 3):
                if data.shape[:-1] != shape:
                    raise ValueError(f"Field '{name}' has mismatched shape")
                flat_data = data.reshape(-1, data.shape[-1])
                comps = data.shape[-1]
            else:
                raise ValueError(
                    f"Field '{name}' must be 2D/3D scalar or 2D/3D vector."
                )

            vtk_array = vtk.util.numpy_support.numpy_to_vtk(
                num_array=flat_data, deep=True, array_type=data_type
            )
            vtk_array.SetName(name)
            vtk_array.SetNumberOfComponents(comps)

            grid.GetPointData().AddArray(vtk_array)

        grid.GetPointData().SetActiveScalars(first_key)

        time_array = vtk.vtkFloatArray()
        time_array.SetName("TIME")
        time_array.SetNumberOfComponents(1)
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, time)
        grid.GetFieldData().AddArray(time_array)

        if domain_size is not None and len(shape) != len(domain_size):
            raise ValueError("Dimension mismatch between data and domain_size")

        Output._coord_setup(shape, domain_size, grid, data_type)
        Output._writer_setup(filename, grid)
