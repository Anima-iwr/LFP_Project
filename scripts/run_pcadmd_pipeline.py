from pcadmd_neural.pipeline import run_pcadmd_pipeline

results = run_pcadmd_pipeline(
    lfp_csv="LFP_data_all_channels.csv",
    max_samples=200000,
    fs=30000,
    window_size=3000,
    step=30,
    latent_dim=8,
)

print("Average KL:", results["metrics"]["avg_kl"])
print("Average Hellinger:", results["metrics"]["avg_hellinger"])
print("Eigenvalues:", results["eigenvalues"])
