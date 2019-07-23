import os
import pytest
from mirapy.data import load_dataset


def test_load_xray_binary_data():
    path = 'test_XRayBinary/'
    filename = 'test.asc'
    
    os.mkdir(path)
    with open(path+filename, 'w') as f:
        f.write('J0352+309 Pulsar      50139.6     0.363137     0.995448      1.54628\n') 
    x, y = load_dataset.load_xray_binary_data(path)
    
    os.remove(path+filename)
    os.rmdir(path)

    assert len(x) == 1 and len(x) == len(y)
