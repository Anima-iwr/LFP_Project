import numpy as np

def create_windows_multichannel(signal_matrix, window_size=3000, step=30):
    X = []
    for i in range(0, len(signal_matrix) - window_size, step):
        window = signal_matrix[i:i + window_size, :]
        X.append(window.flatten())
    return np.array(X)
