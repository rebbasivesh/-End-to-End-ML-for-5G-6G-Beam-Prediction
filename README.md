# End-to-End Machine Learning for 5G/6G Handover & Beam Prediction

This repository contains the complete, production-grade machine learning pipeline for predicting optimal beam indices in 5G/6G millimeter-wave (mmWave) and Terahertz (THz) networks. It is built to leverage Channel State Information (CSI) matrices from ray-tracing wireless propagation models to make rapid, high-accuracy handover decisions.

---

## 📂 Repository Organization

The repository is organized cleanly at the root level, making it easy to navigate, run, and share:

```text
s:\CAPSTON\ (Repository Root)
├── configs/             # Configuration parameters (YAML) for training and models
├── data/                # Dataset loaders and synthetic generator scripts
│   ├── download_data.py # Handles dataset authentication and down-load fallback
│   ├── raw/             # Staged raw CSV datasets
│   └── processed/       # Sanitized and scaled datasets
├── models/              # Saved model binaries (*.pkl, *.pth)
├── notebooks/           # Exploratory Data Analysis (EDA) Jupyter notebooks
├── outputs/             # Primary pipeline plots and model predictions
│   ├── accuracy_curve.png
│   ├── final_blockage.png
│   ├── final_power_heatmap.png
│   ├── training_progress.gif
│   └── metrics.json
├── results/             # Secondary visualization plots
│   └── plots/           # SNR graphs, RSRP/SINR distributions, and confusion matrices
├── src/                 # Reusable source code modules
│   ├── beamforming/     # Antenna array codebook generation & steering vectors
│   ├── data/            # DeepMIMO channel matrix handlers
│   ├── model/           # ML model architecture definitions (RF, XGBoost, PyTorch LSTM)
│   ├── utils/           # Telecommunications metrics calculations (handover success rate)
│   ├── data_preprocessing.py
│   └── feature_engineering.py
├── main.py              # ML pipeline training and evaluation entrypoint
├── requirements.txt     # Python package requirements
├── visualization.py     # Pipeline visualization entrypoint
└── README.md            # Repository document guide (this file)
```

---

## 📊 Datasets Used

The project relies on realistic, ray-tracing derived **Channel State Information (CSI)** matrices rather than simplified ad-hoc signal models:
1. **DeepMIMO CSI Dataset**: Channel state vectors derived from ray-tracing simulations (default scenario: `O1_60` outdoor street canyon). It maps physical user coordinates and multipath propagation variables directly to optimal beam vectors.
2. **Local Fallback Data**: A dataset script generates realistic signal fading, blocker losses, and RSRP/SINR ratios to emulate DeepMIMO scenarios when Kaggle/cloud endpoints are offline.
3. **Data Storage**: Large raw data arrays (`*.npy`) are kept in the ignored `archive/` directory to prevent repository bloat while remaining accessible locally to the training scripts.

---

## 🏆 Results Achieved

The machine learning models were trained and benchmarked for both accuracy (optimal beam prediction) and execution latency (crucial for real-time edge network deployment). 

Below is a comparison of the trained models against standard baselines from telecom literature:

| Model | Beam Prediction Accuracy | Inference Latency (ms) | Deployment Readiness |
| :--- | :---: | :---: | :--- |
| **Random Forest** | 93.15% | 0.03 ms | Ready (Lightweight Edge) |
| **XGBoost** | 93.65% | 0.01 ms | **Best Performance/Latency** |
| **LSTM (Temporal Model)** | 95.60% | 0.02 ms | Ready (Sequence-based tracking) |
| **Hybrid Fusion (Ensemble)** | **96.00%** | 0.05 ms | High Accuracy (Multi-model core) |
| **CNN + RNN (Literature Baseline)**| 90.24% | 0.36 ms | Baseline |

*All results were generated and verified using the active pipeline scripts and saved to `outputs/` and `results/plots/`.*

---

## 🚀 Setup and Execution

### 1. Installation
Ensure Python 3.8+ is installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset
Fetch or compile the DeepMIMO scenario:
```bash
python data/download_data.py
```

### 3. Run Training & Evaluation
To run the primary ML pipeline (train Random Forest, XGBoost, log accuracy, and generate heatmaps):
```bash
python main.py
```

### 4. Generate Visualizations
To produce the beam patterns, blocking simulations, and signal performance plots:
```bash
python visualization.py
```
Outputs are written directly to the `results/plots/` and `outputs/` folders.
