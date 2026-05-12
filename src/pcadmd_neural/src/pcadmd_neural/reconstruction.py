import numpy as np

def reconstruct_multichannel(windows, window_size, step, n_channels, initial_start=0):
    n_windows = windows.shape[0]
    last_start = initial_start + (n_windows - 1) * step
    signal_length = last_start + window_size

    full_signal = np.zeros((signal_length, n_channels))
    counts = np.zeros((signal_length, n_channels))

    for i in range(n_windows):
        start = initial_start + i * step
        end = start + window_size
        full_signal[start:end] += windows[i]
        counts[start:end] += 1

    counts[counts == 0] = 1
    return full_signal / counts
