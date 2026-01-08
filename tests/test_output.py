import os

import numpy as np

from tpb_measurement import output


def test_output():
    data = np.zeros((3, 3))
    output.Output.NumpyToVTK(data, filename="test_output.vtk", time=0.0)
    os.remove("test_output.vtk")
