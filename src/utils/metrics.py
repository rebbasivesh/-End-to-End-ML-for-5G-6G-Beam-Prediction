from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

def calculate_handover_success_rate(y_true, y_pred, margin=1):
    """
    Custom metric: Handover success rate.
    Counts a handover as successful if the predicted beam is exactly the true beam,
    or within a certain acceptable margin (e.g., adjacent beam).
    We assume the beam index represents a rough angular sector, so adjacent beams 
    (distance <= margin) might still give acceptable, albeit not perfect, service.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    differences = np.abs(y_true - y_pred)
    # Treat 0 and 7 as adjacent if it's a full 360 degree 8-beam setup
    # circular_diff = np.minimum(differences, 8 - differences)
    
    successful_handovers = np.sum(differences <= margin)
    return successful_handovers / len(y_true)

def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    hosr = calculate_handover_success_rate(y_true, y_pred, margin=0)  # strict exact match
    hosr_relaxed = calculate_handover_success_rate(y_true, y_pred, margin=1) # relaxed margin

    return {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "handover_success_rate_strict": float(hosr),
        "handover_success_rate_relaxed": float(hosr_relaxed)
    }
