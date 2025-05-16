import numpy as np
from typing import List, Dict, Callable
import math # For log in cross_entropy_loss

# Attempt to import the 'average' function from stats_tools for r_squared
try:
    from stats_tools import average
except ImportError:
    # Define a simple average function as a fallback if stats_tools.average is not available
    # This makes ml_metrics_tools potentially standalone for r_squared if needed,
    # though ideally the dependency is met.
    print("Warning: Could not import 'average' from stats_tools. Using a local fallback for r_squared.")
    def average(numbers: List[float]) -> float:
        if not numbers: return 0.0 # Simplified error handling for fallback
        return sum(numbers) / len(numbers)

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py

def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculates the Mean Squared Error (MSE) between true and predicted values.
    MSE = (1/n) * sum((y_true_i - y_pred_i)^2)

    Args:
        y_true: A list of actual target values (floats).
        y_pred: A list of predicted values (floats). Must be same length as y_true.
    Returns:
        The Mean Squared Error.
    Raises:
        ValueError: If lists are empty or have different lengths.
    """
    if not y_true or not y_pred:
        raise ValueError("Input lists y_true and y_pred cannot be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("Lists y_true and y_pred must have the same length.")
    
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    mse = np.mean((y_true_np - y_pred_np)**2)
    return float(mse)

def r_squared(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculates the R-squared (coefficient of determination) regression score.
    R^2 = 1 - (SS_res / SS_tot)
    SS_res = sum((y_true_i - y_pred_i)^2)
    SS_tot = sum((y_true_i - mean(y_true))^2)

    Args:
        y_true: A list of actual target values (floats).
        y_pred: A list of predicted values (floats). Must be same length as y_true.
    Returns:
        The R-squared score.
    Raises:
        ValueError: If lists are empty, have different lengths, or if SS_tot is zero.
    """
    if not y_true or not y_pred:
        raise ValueError("Input lists y_true and y_pred cannot be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("Lists y_true and y_pred must have the same length.")
    if len(y_true) < 2: # Need at least 2 points for SS_tot to be meaningful if not all points are same
        # Depending on definition, R^2 for <2 points can be undefined or 0.
        # Let's raise error for clarity for now or return a conventional value like 0.0.
        raise ValueError("R-squared requires at least two data points in y_true.")

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    ss_res = np.sum((y_true_np - y_pred_np)**2)
    mean_y_true = average(y_true) # Uses imported or fallback average
    ss_tot = np.sum((y_true_np - mean_y_true)**2)

    if math.isclose(ss_tot, 0.0):
        # If total sum of squares is zero, means all y_true are the same.
        # If ss_res is also zero (perfect prediction), R^2 is 1.
        # Otherwise (ss_res > 0), R^2 is undefined or can be considered 0 or negative infinity.
        # Scikit-learn returns 0.0 if ss_res is also zero, else it can be negative.
        # For simplicity, if ss_tot is 0, perfect fit if ss_res is 0, else not a good fit (return 0 or handle as error)
        return 1.0 if math.isclose(ss_res, 0.0) else 0.0 
        # raise ValueError("Total sum of squares (SS_tot) is zero, R-squared is undefined or model is not useful.")

    r2 = 1 - (ss_res / ss_tot)
    return float(r2)

def cross_entropy_loss(y_true: List[float], y_pred: List[float], epsilon: float = 1e-12) -> float:
    """
    Calculates the cross-entropy loss (log loss) for binary or multi-class classification.
    Assumes y_true contains one-hot encoded values (0 or 1) and y_pred contains probabilities.
    For binary classification, y_true can be [0,1] and y_pred the probability of class 1.
    L = -sum(y_true_i * log(y_pred_i) + (1-y_true_i) * log(1-y_pred_i)) for binary when y_true is 0 or 1.
    More generally for multi-class: L = -sum(y_true_class * log(y_pred_class))
    This implementation focuses on the general case where y_true are labels (0 or 1) and y_pred are probabilities for the positive class.
    For lists of probabilities (e.g. softmax output for y_pred and one-hot for y_true for multiple classes):
    L = -sum over classes (y_true_c * log(y_pred_c))

    This simple version assumes y_true are ground truth labels (0 or 1 for binary, or one-hot vectors for multi-class represented flatly)
    and y_pred are the predicted probabilities for the corresponding positive classes or one-hot encoded predictions.
    The current implementation sums over the elements, assuming each pair (y_true_i, y_pred_i) is meaningful for direct log loss calculation.
    This is suitable for binary cross-entropy where y_true elements are 0 or 1, and y_pred are P(class=1).

    Args:
        y_true: List of ground truth labels (e.g., [1, 0, 1]). Expected to be 0 or 1.
        y_pred: List of predicted probabilities for the positive class (e.g., [0.9, 0.1, 0.8]).
                Values should ideally be in (0,1). Epsilon is used to clip predictions to avoid log(0).
        epsilon: A small float to clip predicted probabilities to [epsilon, 1-epsilon]
                 to prevent log(0) or log(1) issues if predictions are exactly 0 or 1.
    Returns:
        The average cross-entropy loss.
    Raises:
        ValueError: If lists are empty or have different lengths.
    """
    if not y_true or not y_pred:
        raise ValueError("Input lists y_true and y_pred cannot be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("Lists y_true and y_pred must have the same length.")

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # Clip predictions to avoid log(0) error
    y_pred_np = np.clip(y_pred_np, epsilon, 1.0 - epsilon)

    # Binary cross-entropy formula: -(y*log(p) + (1-y)*log(1-p))
    # This assumes y_true elements are 0 or 1.
    loss = -np.sum(y_true_np * np.log(y_pred_np) + (1 - y_true_np) * np.log(1 - y_pred_np))
    
    return float(loss / len(y_true)) # Return mean loss

def confusion_matrix_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Calculates various classification metrics from confusion matrix components.
    (TP: True Positives, TN: True Negatives, FP: False Positives, FN: False Negatives)

    Args:
        tp: Number of True Positives.
        tn: Number of True Negatives.
        fp: Number of False Positives.
        fn: Number of False Negatives.
    Returns:
        A dictionary containing:
        'accuracy': (TP+TN)/(TP+TN+FP+FN)
        'precision': TP/(TP+FP) (Specificity of positive predictions)
        'recall': TP/(TP+FN) (Sensitivity, True Positive Rate)
        'specificity': TN/(TN+FP) (True Negative Rate)
        'f1_score': 2*(Precision*Recall)/(Precision+Recall)
        'positive_likelihood_ratio': Recall / (1 - Specificity)
        'negative_likelihood_ratio': (1 - Recall) / Specificity
        'diagnostic_odds_ratio': PositiveLikelihoodRatio / NegativeLikelihoodRatio
    Raises:
        ValueError: If any input (tp, tn, fp, fn) is negative.
    """
    if any(count < 0 for count in [tp, tn, fp, fn]):
        raise ValueError("Counts for TP, TN, FP, FN cannot be negative.")

    total_population = tp + tn + fp + fn
    if total_population == 0:
        # All metrics would be undefined or 0/0. Return NaNs or default to 0 for simplicity.
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, 
            "specificity": 0.0, "f1_score": 0.0,
            "positive_likelihood_ratio": 0.0,
            "negative_likelihood_ratio": 0.0,
            "diagnostic_odds_ratio": 0.0
        }

    accuracy = (tp + tn) / total_population if total_population > 0 else 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Also Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Likelihood Ratios
    # Positive Likelihood Ratio (LR+): Recall / (1 - Specificity) = TPR / FPR
    fpr = 1 - specificity # False Positive Rate
    lr_plus = recall / fpr if fpr > 0 else float('inf') # or handle appropriately if recall is 0 too
    if recall == 0 and fpr == 0: lr_plus = 0.0 # Convention if no positives and no false positives

    # Negative Likelihood Ratio (LR-): (1 - Recall) / Specificity = FNR / TNR
    fnr = 1 - recall # False Negative Rate
    lr_minus = fnr / specificity if specificity > 0 else float('inf')
    if fnr == 0 and specificity == 0 : lr_minus = 0.0

    # Diagnostic Odds Ratio (DOR): LR+ / LR-
    dor = lr_plus / lr_minus if lr_minus > 0 and lr_minus != float('inf') and lr_plus != float('inf') else float('inf')
    if lr_plus == 0 and lr_minus == 0: dor = 0.0 # if both are 0, implies no info
    if lr_minus == float('inf') and lr_plus != float('inf'): dor = 0.0 # Good sign
    if lr_plus == float('inf') and lr_minus == 0.0: dor = float('inf') # Very good sign


    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1_score),
        "positive_likelihood_ratio": float(lr_plus),
        "negative_likelihood_ratio": float(lr_minus),
        "diagnostic_odds_ratio": float(dor)
    }

def get_ml_metrics_tools() -> List[Callable]:
    """Returns a list of all machine learning metrics tool functions."""
    return [
        mean_squared_error,
        r_squared,
        cross_entropy_loss,
        confusion_matrix_metrics
    ] 