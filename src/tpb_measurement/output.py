import numpy as np
import vtk.util.numpy_support


class Output:
    def __init__(self):
        pass

    @staticmethod
    def NumpyToVTK(
        data,
        filename: str = "test.vtk",
        time: float = 0.0,
        domain_size: np.ndarray = None,
    ):
        data_type = vtk.VTK_FLOAT
        shape = data.shape

        # Flatten the data array so we only to convert the numpy array once
        flat_data_array = data.flatten()
        vtk_data = vtk.util.numpy_support.numpy_to_vtk(
            num_array=flat_data_array, deep=True, array_type=data_type
        )

        # Create the rectilinear grid object
        grid = vtk.vtkRectilinearGrid()

        grid.GetPointData().SetScalars(vtk_data)

        time_array = vtk.vtkFloatArray()
        time_array.SetName("TIME")
        time_array.SetNumberOfComponents(1)
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, time)
        grid.GetFieldData().AddArray(time_array)

        if domain_size is not None and len(shape) != len(domain_size):
            raise ValueError("Dimension mismatch between data and domain_size")

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

        writer = vtk.vtkRectilinearGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.Write()
