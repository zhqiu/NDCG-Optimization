import numpy as np

def precision_at_k(y_pred, y_true, K):
    """
    Calculate Precision@K

    Parameters:
    y_pred (numpy.ndarray): Predicted scores, shape [N, 1]
    y_true (numpy.ndarray): Actual labels, shape [N, 1], each label is either 0 or 1
    K (int): The K value for which to calculate Precision

    Returns:
    float: Computed Precision@K
    """
    # Ensure K does not exceed the number of samples
    K = min(K, y_pred.shape[0])
    
    indices = np.argsort(-y_pred.squeeze())
    top_k_true = y_true[indices[:K]]
    true_positives = np.sum(top_k_true)
    
    precision = true_positives / K
    return precision
