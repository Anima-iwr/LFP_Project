import numpy as np
import pandas as pd

from .windowing import create_windows_multichannel
from .model import PCADMD
from .reconstruction import reconstruct_multichannel
from .metrics import kl_hellinger_per_channel

def run_pcadmd_pipeline(
    lfp_csv,
    max_samples=200000,
    fs=30000,
    window_size=3000,
    step=30,
    latent_dim=8,
):
    lfp_data = pd.read_csv(lfp_csv)
    channels = lfp_data.columns.tolist()

    signals = lfp_data[channels].values.astype(np.float32)[:max_samples]

    X_full = create_windows_multichannel(signals, window_size, step)
    X = X_full[:-1]
    X_next = X_full[1:]

    model = PCADMD(latent_dim=latent_dim)
    model.fit(X, X_next)

    X_pred = model.predict(X)

    X_pred = X_pred.reshape(-1, window_size, len(channels))
    X_next_reshaped = X_next.reshape(-1, window_size, len(channels))

    initial_start = step
    full_original = reconstruct_multichannel(
        X_next_reshaped, window_size, step, len(channels), initial_start
    )

    full_predicted = reconstruct_multichannel(
        X_pred, window_size, step, len(channels), initial_start
    )

    recon_start = step
    recon_length = len(full_original) - recon_start

    original_trim = signals[recon_start:recon_start + recon_length]
    recon_trim = full_predicted[recon_start:recon_start + recon_length]

    metrics = kl_hellinger_per_channel(original_trim, recon_trim)

    time_points = np.arange(recon_length) / fs

    return {
        "model": model,
        "channels": channels,
        "original": original_trim,
        "reconstructed": recon_trim,
        "time": time_points,
        "metrics": metrics,
        "eigenvalues": model.eigenvalues_,
    }
