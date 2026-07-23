import os
import yaml
import numpy as np
import pandas as pd


def generate_synthetic_data(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    n_samples = config["data"].get("n_samples_synthetic", 10000)
    raw_dir = config["data"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    # Generate RSRP: -120 to -60 dBm
    rsrp = np.random.uniform(-120, -60, n_samples)

    # Generate SINR: -10 to 30 dB
    sinr = np.random.uniform(-10, 30, n_samples)

    # Previous beam index (0 to 7)
    prev_beam = np.random.randint(0, 8, n_samples)

    # Simulate current best beam based on RSRP & SINR plus some noise
    # We create artificial "sectors" for the beams depending on RSRP and SINR ranges
    # Just to create a pattern the ML model can learn
    score = (rsrp + 120) / 60 + (sinr + 10) / 40
    # Add dependency on prev_beam to make it sequential-like
    score += (prev_beam / 8.0)

    # Map score to beam index 0-7
    beam_label = np.floor((score / score.max()) * 7.99).astype(int)

    # Some optional pos feature: simulated distance from base station
    distance = np.random.uniform(50, 1000, n_samples)

    df = pd.DataFrame({
        "RSRP": rsrp,
        "SINR": sinr,
        "prev_beam": prev_beam,
        "distance": distance,
        "best_beam": beam_label
    })

    output_path = os.path.join(raw_dir, "dataset.csv")
    df.to_csv(output_path, index=False)
    print(
        f"Generated synthetic data with {n_samples} samples at {output_path}")


def download_dataset():
    # Attempt kaggle download, but do not create synthetic signal values.
    try:
        if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')) and 'KAGGLE_USERNAME' not in os.environ:
            raise Exception(
                "Kaggle not authenticated. DeepMIMO dataset download cannot proceed.")
        import kaggle
        print("Attempting to download DeepMIMO dataset from Kaggle...")
        kaggle.api.authenticate()
        print("Kaggle authentication successful. Please implement dataset retrieval logic here.")
    except BaseException as e:
        error_msg = (
            f"DeepMIMO dataset download unavailable ({e}). "
            "This project uses only DeepMIMO CSI and does not fall back to synthetic signal generation. "
            "Please provide the DeepMIMO dataset at the configured location."
        )
        print(error_msg)
        raise RuntimeError(error_msg)


if __name__ == "__main__":
    download_dataset()
