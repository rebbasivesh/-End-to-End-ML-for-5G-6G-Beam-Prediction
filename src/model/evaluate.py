import torch
import numpy as np

def evaluate_model(model, X_test, y_test):
    print("Evaluating Final Model performance on Test Dataset...")
    if hasattr(model, 'predict'):
        # Scikit-learn model
        y_pred = model.predict(X_test)
    else:
        # PyTorch model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        
        # Needs to be a tensor matching device schema
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs.data, 1)
            
        y_pred = predicted.cpu().numpy()
        
    # Final strictly independent accuracy check
    correct = (y_pred == y_test).sum()
    accuracy = correct / len(y_test)
    print(f"Final Offline Model Benchmark Accuracy: {accuracy * 100:.2f}%")
        
    return y_pred
