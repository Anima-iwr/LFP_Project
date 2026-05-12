import numpy as np
from pcadmd_neural.windowing import create_windows_multichannel

def test_create_windows_multichannel():
    x = np.random.randn(100, 2)
    windows = create_windows_multichannel(x, window_size=10, step=5)
    assert windows.shape[1] == 20
