import os
import sys
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.deepmimo_loader import load_deepmimo_data
from src.beamforming.beam_selector import extract_csi_beam_power, select_best_beam


def build_offline_pipeline(config_path="configs/config.yaml"):
    """
    Builds the ML dataset directly from DeepMIMO ray-tracing CSI.
    """
    print("\n=== Processing DeepMIMO Ray-Tracing Dataset ===")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = load_deepmimo_data(
        dataset_folder=config["data"].get(
            "deepmimo_dataset_folder", r"S:\CAPSTON\DeepMIMO_Dataset"),
        scenario=config["data"].get("scenario", "O1_60"),
        user_row_first=config["data"].get("user_row_first", 1),
        user_row_last=config["data"].get("user_row_last", 1000),
        fallback_path=config["data"].get(
            "fallback_path", r"S:\CAPSTON\archive\H_train_doppler.npy"),
    )

    raw_csi = dataset["csi"]
    positions = dataset.get("positions")
    path_info = dataset.get("paths")

    print(f"[Pipeline] Raw CSI shape: {raw_csi.shape}")
    assert raw_csi.ndim in (4, 5), "DeepMIMO CSI must be 4D or 5D array."
    assert raw_csi.shape[-1] > 0, "CSI Tx dimension must be present."

    if positions is not None:
        try:
            print(f"[Pipeline] User positions shape: {positions.shape}")
        except Exception:
            print("[Pipeline] User positions loaded.")
    if path_info is not None:
        print("[Pipeline] Path-level metadata available from DeepMIMO.")

    spatial_features, raw_power_map = extract_csi_beam_power(
        raw_csi, return_raw=True)
    y = select_best_beam(spatial_features)
    X = spatial_features

    test_size = config["training"].get("test_size", 0.2)
    random_state = config["training"].get("random_state", 42)

    print("[Pipeline] Splitting data into Train/Test partitions...")
    X_train, X_test, raw_power_train, raw_power_test, y_train, y_test = train_test_split(
        X, raw_power_map, y, test_size=test_size, random_state=random_state)

    print(
        f"Final Pipeline Bounds - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(
        f"Final Pipeline Bounds - X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test, raw_power_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = build_offline_pipeline()
