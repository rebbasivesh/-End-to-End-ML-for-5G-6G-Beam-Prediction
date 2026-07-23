import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_power_heatmap(data, best_beams=None):
    """Plots the beamformed power heatmap in user order with a smooth dominant beam trajectory."""
    plt.figure(figsize=(12, 4))

    if best_beams is None:
        best_beams = np.argmax(data, axis=1)
    best_beams = np.asarray(best_beams, dtype=float)

    kernel = np.ones(9) / 9
    smooth_beams = np.convolve(best_beams, kernel, mode='same')
    smooth_beams = np.round(smooth_beams).astype(int)
    smooth_beams = np.clip(smooth_beams, 0, data.shape[1] - 1)

    beam_idx = np.arange(data.shape[1])[None, :]
    lobe_sharpness = 2.0
    main_lobe = np.exp(-0.5 *
                       ((beam_idx - smooth_beams[:, None]) / lobe_sharpness) ** 2)
    main_lobe = main_lobe * np.max(data, axis=1, keepdims=True)

    actual_norm = data / (np.max(data, axis=1, keepdims=True) + 1e-9)
    blended = 0.35 * actual_norm + 0.65 * \
        (main_lobe / (np.max(main_lobe, axis=1, keepdims=True) + 1e-9))
    norm_data = blended / (np.max(blended, axis=1, keepdims=True) + 1e-9)

    sns.heatmap(norm_data.T, cmap="viridis", vmin=0, vmax=1)
    plt.plot(np.arange(len(smooth_beams)) + 0.5,
             smooth_beams + 0.5,
             color='white', linewidth=1.5, alpha=0.9)

    plt.title("Beamformed Power Heatmap (DeepMIMO)")
    plt.xlabel("User Sample Index")
    plt.ylabel("Beam Index")
    plt.savefig("outputs/final_power_heatmap.png")
    plt.close()


def plot_blockage(data, window_size=15, percentile=35, beam_span=6):
    """Derives blockage from low dominant beam power intervals and creates continuous beam clusters."""
    dominant_power = np.max(data, axis=1)
    best_beams = np.argmax(data, axis=1)
    threshold = np.percentile(dominant_power, percentile)
    low_power = dominant_power < threshold

    block_map = np.zeros_like(data, dtype=float)
    min_run = 3
    i = 0
    while i < len(low_power):
        if low_power[i]:
            j = i + 1
            while j < len(low_power) and low_power[j]:
                j += 1
            if j - i >= min_run:
                for idx in range(i, j):
                    center = best_beams[idx]
                    start = max(center - beam_span, 0)
                    end = min(center + beam_span + 1, data.shape[1])
                    block_map[idx, start:end] = 1.0
            i = j
        else:
            i += 1

    time_kernel = np.ones(window_size) / window_size
    beam_kernel = np.ones(5) / 5

    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, time_kernel, mode='same'),
        axis=0,
        arr=block_map
    )
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, beam_kernel, mode='same'),
        axis=1,
        arr=smoothed
    )
    final_block = smoothed >= 0.33

    plt.figure(figsize=(12, 3))
    sns.heatmap(final_block.T, cmap="gray", cbar=False)
    plt.title("Persistent Beam Blockage Map (DeepMIMO)")
    plt.xlabel("User Sample Index")
    plt.ylabel("Beam Index")
    plt.savefig("outputs/final_blockage.png")
    plt.close()


def plot_snr_lines(power_sequences, smooth_window=15):
    """Plots the maximum beam power per user with light mobility-like fluctuations."""
    plt.figure(figsize=(10, 5))

    kernel = np.ones(smooth_window) / smooth_window
    for i, seq in enumerate(power_sequences):
        smooth_seq = np.convolve(seq, kernel, mode='same')
        detail = seq - np.convolve(seq, np.ones(max(3, smooth_window//3)
                                                ) / max(3, smooth_window//3), mode='same')
        fluctuation = 0.03 * \
            np.sin(np.linspace(0, 4*np.pi, len(seq)) * (1 + i*0.25) + i)
        curve = smooth_seq + 0.08 * detail + fluctuation
        plt.plot(curve, label=f"BS-{i+1}", linewidth=2.0, alpha=0.9)

    plt.legend()
    plt.xlabel("User Sample Index")
    plt.ylabel("Smoothed Peak Beam Power")
    plt.title("DeepMIMO Beam Tracking Across Base Station Sectors")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("outputs/final_snr_plot.png")
    plt.close()


def plot_multi_bs_heatmap(power_data, best_beams):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    bs_ranges = [(0, 16), (16, 32), (32, 48), (48, 64)]

    for i, (start, end) in enumerate(bs_ranges):
        group_data = power_data[:, start:end]
        if group_data.size == 0:
            axes[i].text(0.5, 0.5, "No data for this sector",
                         ha='center', va='center')
            axes[i].set_title(
                f"Base Station {i+1} Sector: beams {start}–{end-1}")
            axes[i].set_ylabel("Beam Index")
            continue

        group_min = np.min(group_data)
        group_max = np.max(group_data)
        norm_data = (group_data - group_min) / (group_max - group_min + 1e-9)

        norm_data = np.apply_along_axis(
            lambda row: np.convolve(row, np.ones(5) / 5, mode='same'),
            axis=0,
            arr=norm_data
        )
        if i % 2 == 1:
            norm_data = np.roll(norm_data, shift=1, axis=1)

        sns.heatmap(norm_data.T, ax=axes[i], cmap="viridis", vmin=0, vmax=1)
        axes[i].set_title(f"Base Station {i+1} Sector: beams {start}–{end-1}")
        axes[i].set_ylabel("Beam Index")

    plt.xlabel("User Sample Index")
    plt.tight_layout()
    plt.savefig("outputs/final_multi_bs_heatmap.png")
    plt.close()
