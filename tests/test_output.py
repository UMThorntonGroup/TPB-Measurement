import numpy as np
import vtk.util.numpy_support

from tpb_measurement import output


def test_output(tmp_path):
    data = np.zeros((3, 3))
    fields = {"field": data}
    filename = tmp_path / "test_output.vtr"

    output.Output.numpy_to_rectilinear_vtk(
        fields, filename=str(filename), time=0.0
    )
    assert filename.exists()


def test_output_contains_field(tmp_path):
    data = np.zeros((3, 3))
    fields = {"temperature": data}

    filename = tmp_path / "test_output.vtr"

    output.Output.numpy_to_rectilinear_vtk(
        fields,
        filename=str(filename),
        time=1.23,
    )

    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(str(filename))
    reader.Update()

    grid = reader.GetOutput()

    # Grid dimensions
    assert grid.GetDimensions() == (3, 3, 1)

    # Field exists
    point_data = grid.GetPointData()
    assert point_data.HasArray("temperature")


def test_multiple_fields(tmp_path):
    data1 = np.zeros((3, 3))
    data2 = np.ones((3, 3))

    fields = {
        "zeros": data1,
        "ones": data2,
    }

    filename = tmp_path / "multi_field.vtr"

    output.Output.numpy_to_rectilinear_vtk(
        fields,
        filename=str(filename),
        time=0.0,
    )

    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(str(filename))
    reader.Update()

    grid = reader.GetOutput()
    pd = grid.GetPointData()

    assert pd.HasArray("zeros")
    assert pd.HasArray("ones")
    assert pd.GetNumberOfArrays() == 2
