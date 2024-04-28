import numpy as np
from sklearn.metrics import average_precision_score


def map_at_k(y_true, y_pred, k, num_tasks=128):
    """
    Calculate the mean Average Precision at k (mAP@k) for 128 tasks.
    
    Parameters:
    - y_true: a numpy.ndarray of shape [N, 128], elements are 0 (negative) or 1 (positive)
    - y_pred: a numpy.ndarray of shape [N, 128], elements are scores indicating the prediction confidence
    
    Returns:
    - mAP@k: mean of the Average Precision at k across all tasks
    """
    # Initialize an empty list to store AP values for each task
    ap_scores = []
    
    # Number of samples
    N = y_true.shape[0]
    
    # Iterate over each task
    for i in range(num_tasks):  # Assuming there are always 128 tasks
        # Get the true labels and predicted scores for the task
        true_labels = y_true[:, i]
        scores = y_pred[:, i]
        
        # Sort the samples by descending predicted score
        sorted_indices = np.argsort(-scores)
        sorted_true_labels = true_labels[sorted_indices]
        
        # Consider only the top k samples
        if k > N:
            top_k_true_labels = sorted_true_labels
        else:
            top_k_true_labels = sorted_true_labels[:k]
        
        # Compute Average Precision at k for the task
        # Here we assume there's at least one positive example in the top k
        if np.sum(top_k_true_labels) == 0:
            ap_score = 0.0
        else:
            ap_score = average_precision_score(top_k_true_labels, scores[sorted_indices][:k])
        
        # Append the AP score to the list
        ap_scores.append(ap_score)
    
    # Calculate mean AP across all tasks
    mean_ap = np.mean(ap_scores)
    
    return mean_ap

