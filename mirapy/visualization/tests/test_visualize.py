from mirapy.visualization import visualize
import numpy as np
import pytest


def test_visualize_2d():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([1, 2])
    visualize.visualize_2d(a, b)

def test_visualize_3d():
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    b = np.array([1, 2, 3])
    visualize.visualize_3d(a, b)
