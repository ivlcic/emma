import logging
import numpy as np
from argparse import ArgumentParser

from sklearn.metrics import ndcg_score, dcg_score
from sympy import denom

from ..core.args import CommonArguments
from ..core.metrics import MetricsAtK

logger = logging.getLogger('tests.ir_metrics')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_in_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-i', '--data_out_dir'))


from typing import Optional, Dict
import numpy as np
from sklearn.metrics import ndcg_score


def r_precision_at_k(y_true, y_pred, k):
    """
    Compute R-Precision@K for multiple samples.

    Args:
    y_true: 2D array of true relevance labels, shape (n_samples, n_labels)
    y_pred: 2D array of predicted scores or probabilities, shape (n_samples, n_labels)
    k: The number of top items to consider for Recall@K

    Returns:
    Array of R-Precision@K scores for each sample and the mean R-Precision@K.
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


def dcg_score1(y_true, y_score, k=None):
    """Compute Discounted Cumulative Gain (DCG)"""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score1(y_true, y_score, k=None):
    """Compute Normalized Discounted Cumulative Gain (nDCG)"""
    dcg = dcg_score1(y_true, y_score, k)
    ideal_dcg = dcg_score1(y_true, y_true, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0


def recall_at_k(y_true, y_pred, k):
    """
    Compute Recall@K for multiple samples.

    Args:
    y_true: 2D array of true relevance labels, shape (n_samples, n_labels)
    y_pred: 2D array of predicted scores or probabilities, shape (n_samples, n_labels)
    k: The number of top items to consider for Recall@K

    Returns:
    Array of Recall@K scores for each sample and the mean Recall@K.
    """
    n_samples = y_true.shape[0]
    recalls = np.zeros(n_samples)

    for i in range(n_samples):
        # Get top K indices for this sample
        top_k_indices = np.argsort(y_pred[i])[::-1][:k]

        # Count relevant items in top K
        relevant_in_k = np.sum(y_true[i][top_k_indices])
        # Total number of relevant items for this sample
        total_relevant = np.sum(y_true[i])

        # Compute Recall@K for this sample
        if total_relevant > 0:
            recalls[i] = relevant_in_k / total_relevant
        else:
            recalls[i] = 0  # If no relevant items, recall is 0

    # Compute mean Recall@K
    mean_recall = np.mean(recalls)
    return recalls, mean_recall


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


def ir_metrics_ndcg(args) -> int:
    # Example data
    y_true = np.array([1, 0, 1, 1])
    y_score = np.array([0.1, 0.1, 0.8, 0.7])

    # Compute nDCG
    ndcg1 = ndcg_score1(y_true, y_score)
    print(f"nDCG1 score: {ndcg1}")
    ndcg = ndcg_score(np.array([y_true]), np.array([y_score]), ignore_ties=True)
    print(f"nDCG score: {ndcg}")


    # Let's break down the calculation
    print("\nStep-by-step calculation:")

    # Step 1: Sort scores in descending order
    order = np.argsort(y_score)[::-1]
    print(f"Sorted order: {order}")

    # Step 2: Reorder true labels based on predicted scores
    y_true_sorted = y_true[order]
    print(f"Reordered true labels: {y_true_sorted}")

    # Step 3: Calculate gains
    gains = 2 ** y_true_sorted - 1
    print(f"Gains: {gains}")

    # Step 4: Calculate discounts
    neki = np.arange(len(y_true))
    discounts = np.log2(neki + 2)
    print(f"Discounts: {discounts}")

    # Step 5: Calculate DCG
    dcg = np.sum(gains / discounts)
    print(f"DCG: {dcg}")

    # Step 6: Calculate ideal DCG
    ideal_order = np.argsort(y_true)[::-1]
    y_true_ideal = y_true[ideal_order]
    ideal_gains = 2 ** y_true_ideal - 1
    ideal_dcg = np.sum(ideal_gains / discounts)
    print(f"Ideal DCG: {ideal_dcg}")

    return 0


def ir_metrics_ndcg_at(args) -> int:
    # Example data
    y_true = np.array([0, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.1, 0.8, 0.7, 0.2])

    k = 4
    # Compute nDCG
    ndcg1 = ndcg_score1(y_true, y_score, k)
    print(f"nDCG1 score: {ndcg1}")
    ndcg = ndcg_score(np.array([y_true]), np.array([y_score]), k=k, ignore_ties=True)
    dcg = dcg_score(np.array([y_true]), np.array([y_score]), k=k, ignore_ties=True)
    print(f"nDCG score: {ndcg} , dcg {dcg}")

    print("\nStep-by-step calculation:")

    # Step 1: Sort scores in descending order
    order = np.argsort(y_score)[::-1]
    print(f"Sorted order: {order}")

    # Step 2: Reorder true labels based on predicted scores
    y_true_ordered = y_true[order][:k]

    rank = np.arange(len(y_true[:k])) + 2  # +2 = 1 from the equation and 1 since we have zero based indexing
    print(f"Reordered ground truth labels: {y_true_ordered} with rank {rank}")
    y_log_sorted = np.log2(rank)
    print(f"Reordered log {y_log_sorted} of indices + 1 starting from one in reverse order {(order + 2)}")

    dcg = np.sum(y_true_ordered / y_log_sorted)
    print(f"Quotient sum ordered true labels over log of an index: {dcg}")
    num_relevant = np.sum(y_true)
    min_of = min(num_relevant, k)
    denom_vect = np.arange(len(y_true[:min_of])) + 2
    print(f"Quotient denominator to sum: {denom_vect}")
    denom_vect = 1 / np.log2(denom_vect)
    print(f"1 over Log Quotient denominator to sum: {denom_vect}")
    idcg = np.sum(denom_vect)
    print(f"Ideal DCG: {idcg}")
    print(f"nDCG: {(dcg / idcg)}")

    return 0
