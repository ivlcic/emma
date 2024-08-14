from typing import Optional

import numpy as np


def r_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: Optional[int]):
    """
    Compute R-Precision@K for multiple samples.

    Args:
    y_true: 2D array of true relevance labels, shape (n_samples, n_labels)
    y_pred: 2D array of predicted scores or probabilities, shape (n_samples, n_labels)
    k: The number of top items to consider for Recall@K

    Returns:
    Array of Recall@K scores for each sample and the mean Recall@K.
    """
    n_samples = y_true.shape[0]
    r_precisions = np.zeros(n_samples)

    for i in range(n_samples):
        # Total number of relevant items for this sample
        total_relevant = np.sum(y_true[i])
        if k is not None:
            total_relevant = min(total_relevant, k)
        else:
            k = total_relevant  # here it becomes recall

        # Get top K indices for this sample
        top_k_indices = np.argsort(y_pred[i])[::-1][:k]

        # Count relevant items in top K
        relevant_in_k = np.sum(y_true[i][top_k_indices])

        # Compute Recall@K for this sample
        if total_relevant > 0:
            r_precisions[i] = relevant_in_k / total_relevant
        else:
            r_precisions[i] = 0  # If no relevant items, recall is 0

    # Compute mean Recall@K
    mean_r_precision = np.mean(r_precisions)
    return mean_r_precision, r_precisions
