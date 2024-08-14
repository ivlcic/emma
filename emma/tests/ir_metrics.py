import logging
import numpy as np
from argparse import ArgumentParser

from ..core.args import CommonArguments

logger = logging.getLogger('tests.llm')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_in_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-i', '--data_out_dir'))


def r_precision_at_k(y_true, y_pred, k):
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
        print(f'r:{total_relevant}')
        total_relevant = min(total_relevant, k)
        print(f'min(r,k):{total_relevant}')
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


def ir_metrics_recall(args) -> int:
    # Example usage
    y_true = np.array([
        [1, 0, 1, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 1],
        [0, 1, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1]
    ])
    y_pred = np.array([
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.3, 0.6, 0.2, 0.7, 0.1, 0.9, 0.5],
        [0.3, 0.6, 0.2, 0.7, 0.1, 0.9, 0.9]
    ])

    # Compute Recall@3 and Recall@5
    k_values = [3, 5]

    for k in k_values:
        recalls, mean_recall = recall_at_k(y_true, y_pred, k)
        print(f"\nRecall@{k}:")
        for i, recall in enumerate(recalls):
            print(f"Sample {i + 1}: {recall:.4f}")
        print(f"Mean Recall@{k}: {mean_recall:.4f}")
    return 0