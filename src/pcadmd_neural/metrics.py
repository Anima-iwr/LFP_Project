import numpy as np
from scipy.stats import entropy

def kl_hellinger_per_channel(original, reconstructed, bins=100, epsilon=1e-10):
    kl_divs = []
    hellingers = []

    for ch in range(original.shape[1]):
        hist_orig, bin_edges = np.histogram(original[:, ch], bins=bins, density=True)
        hist_pred, _ = np.histogram(reconstructed[:, ch], bins=bin_edges, density=True)

        hist_orig = (hist_orig + epsilon) / (hist_orig + epsilon).sum()
        hist_pred = (hist_pred + epsilon) / (hist_pred + epsilon).sum()

        kl_divs.append(entropy(hist_orig, hist_pred))
        hellingers.append(np.sqrt(1 - np.sum(np.sqrt(hist_orig * hist_pred))))

    return {
        "kl_per_channel": kl_divs,
        "hellinger_per_channel": hellingers,
        "avg_kl": float(np.mean(kl_divs)),
        "avg_hellinger": float(np.mean(hellingers)),
    }
