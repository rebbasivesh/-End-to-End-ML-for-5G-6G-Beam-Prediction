import os
import numpy as np


def _extract_user_positions(dataset):
    try:
        bs_data = dataset[0] if isinstance(
            dataset, (list, tuple)) and len(dataset) > 0 else dataset
        user_info = bs_data.get("user", {})
        if "position" in user_info:
            return np.asarray(user_info["position"])
        if "location" in user_info:
            return np.asarray(user_info["location"])
        if "user_position" in user_info:
            return np.asarray(user_info["user_position"])
    except Exception:
        pass
    return None


def _extract_path_info(dataset):
    try:
        bs_data = dataset[0] if isinstance(
            dataset, (list, tuple)) and len(dataset) > 0 else dataset
        if "paths" in bs_data:
            return bs_data["paths"]
    except Exception:
        pass
    return None


def load_deepmimo_data(dataset_folder=r"S:\CAPSTON\DeepMIMO_Dataset",
                       scenario="O1_60",
                       user_row_first=1,
                       user_row_last=1000,
                       fallback_path=r"S:\CAPSTON\archive\H_train_doppler.npy"):
    """
    Loads DeepMIMO CSI and metadata.
    Uses ray-tracing data when available, otherwise loads a pre-compiled DeepMIMO CSI file.
    """
    print("\n[DeepMIMO Loader] Initializing physically accurate DeepMIMO channel generation...")
    result = {"csi": None, "positions": None, "paths": None}

    try:
        import DeepMIMO
        parameters = DeepMIMO.default_params()
        parameters["dataset_folder"] = dataset_folder
        parameters["scenario"] = scenario
        parameters["active_BS"] = np.array([1])
        parameters["bs_antenna"]["x"] = 1
        parameters["bs_antenna"]["y"] = 8
        parameters["bs_antenna"]["z"] = 8
        parameters["ue_antenna"]["x"] = 1
        parameters["ue_antenna"]["y"] = 1
        parameters["ue_antenna"]["z"] = 1
        parameters["user_row_first"] = user_row_first
        parameters["user_row_last"] = user_row_last

        dataset = DeepMIMO.generate_data(parameters)
        print(
            "[DeepMIMO Loader] Successfully generated CSI from ray-tracing parameters.")

        bs_data = dataset[0] if isinstance(
            dataset, (list, tuple)) and len(dataset) > 0 else dataset
        result["csi"] = bs_data["user"]["channel"]
        result["positions"] = _extract_user_positions(dataset)
        result["paths"] = _extract_path_info(dataset)

        if result["positions"] is None:
            print(
                "[DeepMIMO Loader] User position metadata not found in generated scenario.")
        if result["paths"] is None:
            print("[DeepMIMO Loader] Path metadata not found in generated scenario.")

        return result
    except Exception as e:
        print(f"[DeepMIMO Loader] Ray-tracing generation unavailable: {e}")
        print(
            f"[DeepMIMO Loader] Falling back to pre-compiled DeepMIMO CSI matrix at {fallback_path} ...")

        if not os.path.exists(fallback_path):
            raise FileNotFoundError(
                f"CRITICAL ERROR: Pre-compiled DeepMIMO fallback not found at {fallback_path}. "
                "Please download the dataset or point `fallback_path` to a valid DeepMIMO file."
            )

        raw_data = np.load(fallback_path, mmap_mode='r')
        result["csi"] = raw_data
        return result


if __name__ == "__main__":
    data = load_deepmimo_data()
    print(f"Loaded CSI Shape: {data['csi'].shape}")
    print(f"User positions: {data['positions']}")
    print(f"Path info present: {data['paths'] is not None}")
