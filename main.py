from src.data_preprocessing import build_offline_pipeline
from src.model.train import multi_epoch_train, run_optuna_optimization
from src.model.evaluate import evaluate_model
from visualization import plot_results
import os


def main():
    print("\n--- 1/2/3. Dataset Handling & Preprocessing ---")
    # This securely performs load_data -> preprocess_data -> create_labels internally
    X_train, y_train, X_test, y_test, raw_power_test = build_offline_pipeline()

    print("\n--- 3.5. Hyperparameter Tuning (Optuna) ---")
    # Limiting to 5 trials conservatively for demonstration and runtime efficiency
    best_params = run_optuna_optimization(
        X_train, y_train, X_test, y_test, num_classes=64, n_trials=5)

    print("\n--- 4/5. Model Selection & Multi-Epoch Training ---")
    num_epochs = 25
    experiment_results, trained_model = multi_epoch_train(
        X_train, y_train, X_test, y_test,
        raw_power_test=raw_power_test,
        num_epochs=num_epochs,
        best_params=best_params,
        enable_visualization=True
    )

    print("\n--- 6. Evaluation ---")
    y_pred = evaluate_model(trained_model, X_test, y_test)

    print("\n--- 7/8/9. Visualization & Logging ---")
    plot_results(experiment_results, y_test, y_pred)

    print("\n--- Pipeline Execution Complete! View graphs in outputs/ ---")


if __name__ == "__main__":
    main()
