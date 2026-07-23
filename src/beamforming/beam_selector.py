import numpy as np


def build_beam_codebook(num_tx, num_beams=None):
    """
    Builds a uniform linear array beam codebook in the angular domain.
    Each column is a steering vector for a candidate transmit beam.
    """
    if num_beams is None:
        num_beams = num_tx

    antenna_idx = np.arange(num_tx)
    angles = np.linspace(-np.pi / 2, np.pi / 2, num_beams)
    codebook = np.exp(-1j * np.pi * np.outer(antenna_idx,
                      np.sin(angles))) / np.sqrt(num_tx)
    return codebook.astype(np.complex64)


def _convert_to_complex(raw_chunk):
    if raw_chunk.ndim == 5 and raw_chunk.shape[2] == 2:
        real_part = raw_chunk[..., 0, :, :]
        imag_part = raw_chunk[..., 1, :, :]
        return real_part + 1j * imag_part
    return raw_chunk


def extract_csi_beam_power(raw_data, codebook=None, chunk_size=500, return_raw=False):
    """
    Converts DeepMIMO CSI into a normalized beam power map using a beam codebook.
    This applies beamforming vectors to the channel and derives per-beam received power.
    """
    print(
        f"[Beam Selector] Extracting beam-level CSI power features in chunks of {chunk_size}...")
    num_users = raw_data.shape[0]
    beam_power_chunks = []

    if raw_data.ndim not in (4, 5):
        raise ValueError(
            "Unsupported CSI shape: expected 4D or 5D DeepMIMO CSI array.")

    for start in range(0, num_users, chunk_size):
        end = min(start + chunk_size, num_users)
        chunk = raw_data[start:end]
        complex_chunk = _convert_to_complex(chunk).astype(np.complex64)

        num_tx = complex_chunk.shape[-1]
        if codebook is None:
            codebook = build_beam_codebook(num_tx)

        # Apply beamforming on each subcarrier and receive antenna.
        # complex_chunk: (batch, subcarriers, rx, tx)
        # codebook: (tx, beams)
        beam_outputs = np.tensordot(complex_chunk, codebook, axes=([3], [0]))
        power_per_rx = np.abs(beam_outputs) ** 2
        power = np.mean(power_per_rx, axis=(1, 2))

        beam_power_chunks.append(power.astype(np.float32))

    beam_power_map = np.concatenate(beam_power_chunks, axis=0)
    raw_power_map = beam_power_map.copy()
    max_per_user = np.max(beam_power_map, axis=1, keepdims=True) + 1e-9
    normalized_features = beam_power_map / max_per_user

    if return_raw:
        return normalized_features, raw_power_map
    return normalized_features


def select_best_beam(spatial_features):
    """
    Selects the optimal beam index from the computed beam power map.
    """
    print("[Beam Selector] Selecting optimal beam from beam power map (Ground Truth)...")
    return np.argmax(spatial_features, axis=1)
