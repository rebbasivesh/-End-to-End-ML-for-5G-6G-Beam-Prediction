import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_results(experiment_results, y_test=None, y_pred=None, out_dir="results/plots"):
    """
    Generates required Multi-Epoch visualizations:
    1. Accuracy vs Epochs
    2. Train vs Test Accuracy Curve
    3. Confusion Matrix
    """
    os.makedirs(out_dir, exist_ok=True)
    print("Generating Experiment Visualizations...")

    epochs = experiment_results["epochs"]
    train_acc = experiment_results["train_acc"]
    test_acc = experiment_results["test_acc"]

    # 1. Accuracy vs Epochs (Test Accuracy usually represents the baseline model metric)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, test_acc, marker='o', linestyle='-', color='b')
    plt.title("Model Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy_vs_epochs.png"), bbox_inches='tight')
    plt.close()

    # 2. Train vs Test Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, marker='o', linestyle='-', color='r', label="Train Accuracy")
    plt.plot(epochs, test_acc, marker='s', linestyle='--', color='b', label="Test Accuracy")
    plt.title("Train vs Test Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "train_vs_test_accuracy.png"), bbox_inches='tight')
    plt.close()

    # 3. Confusion Matrix (requires y_test, y_pred from the final model evaluation)
    if y_test is not None and y_pred is not None:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        # Log scale coloring helps if there are massive imbalances, but pure numeric is fine
        sns.heatmap(cm, cmap="Blues", cbar=True)
        plt.title("Prediction Confusion Matrix")
        plt.ylabel("True Beam Index")
        plt.xlabel("Predicted Beam Index")
        plt.savefig(os.path.join(out_dir, "confusion_matrix_final.png"), bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved successfully to {out_dir}/")
