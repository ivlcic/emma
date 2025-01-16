import json
import os
from typing import Optional, Literal, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, precision_score, recall_score, f1_score, accuracy_score, hamming_loss

from .wandb import send_metrics


class MetricsAtK:

    def __init__(self, y_true: np.ndarray, y_prob: np.ndarray, k: Optional[int] = None):
        n_samples = y_true.shape[0]
        self.mean_r_precision = 0
        self.mean_recall = 0
        self.mean_precision = 0
        self.ndcg = 0
        self.k = k
        self.r_precisions = np.zeros(n_samples)
        self.recalls = np.zeros(n_samples)
        self.precisions = np.zeros(n_samples)

        for i in range(n_samples):
            y_true[i][np.isnan(y_true[i])] = 0
            y_prob[i][np.isnan(y_prob[i])] = 0
            # Total number of relevant items for this sample
            total_relevant = np.sum(y_true[i])
            if k is not None:
                total_relevant_at_k = min(total_relevant, k)
            else:
                k = int(total_relevant)  # here it becomes recall
                total_relevant_at_k = total_relevant

            # Get top K indices for this sample
            top_k_indices = np.argsort(y_prob[i])[::-1][:k]

            # Count relevant items in top K
            relevant_in_k = np.sum(y_true[i][top_k_indices])

            # Compute metrics for this sample
            self.r_precisions[i] = relevant_in_k / total_relevant_at_k if total_relevant_at_k > 0 else 0
            self.recalls[i] = relevant_in_k / total_relevant if total_relevant > 0 else 0
            self.precisions[i] = relevant_in_k / k if k > 0 else 0

        self.mean_r_precision = np.mean(self.r_precisions)
        self.mean_recall = np.mean(self.recalls)
        self.mean_precision = np.mean(self.precisions)
        self.ndcg = ndcg_score(y_true, y_prob, k=self.k)

    def todict(self, prefix: str = '') -> Dict[str, float]:
        suffix = f'@{self.k}' if self.k is not None else ''
        return {
            f'{prefix}r-p{suffix}': self.mean_r_precision,
            f'{prefix}p{suffix}': self.mean_precision,
            f'{prefix}r{suffix}': self.mean_recall,
            f'{prefix}ndcg{suffix}': self.ndcg,
        }


class Metrics:

    def __init__(self, model_name: str, prob_type: Literal['multilabel', 'multiclass', 'binary'] = 'multilabel',
                 avg_k: Optional[int] = None):
        self.log_epochs = []
        self.prob_type = prob_type
        self.model_name = model_name
        self.k_values = [1, 3, 5, 7, 9]
        self.avg_k = avg_k

    # noinspection DuplicatedCode
    def __call__(self, y_true: np.ndarray, y_prob: np.ndarray, prefix: str = '', prob_threshold: float = 0.5):
        if self.prob_type == 'multilabel':
            y_pred = (y_prob > prob_threshold).astype(np.float32)
        else:
            y_pred = np.argmax(y_prob, axis=-1)
        metric = {}
        for average_type in ['micro', 'macro', 'weighted']:
            if self.prob_type == 'binary' and not average_type == 'macro':
                continue
            p = precision_score(y_true, y_pred, average=average_type)
            r = recall_score(y_true, y_pred, average=average_type)
            f1 = f1_score(y_true, y_pred, average=average_type)
            metric[f'{prefix}{average_type}.f1'] = f1
            metric[f'{prefix}{average_type}.p'] = p
            metric[f'{prefix}{average_type}.r'] = r
        metric[f'{prefix}acc'] = accuracy_score(y_true, y_pred)
        if self.prob_type == 'multilabel':
            for k in self.k_values:
                metric = metric | MetricsAtK(y_true, y_prob, k).todict(prefix)
            metric = metric | MetricsAtK(y_true, y_prob).todict(prefix)
            metric[f'{prefix}hamming_loss'] = hamming_loss(y_true, y_pred)
        self.log_epochs.append(metric)
        return metric

    def dump(self, result_path: str, meta_data: Optional[Dict[str, Any]], task: Optional[Any] = None,
             multiplier: int = 1):
        mdf = pd.DataFrame(self.log_epochs)
        mdf_file = os.path.join(result_path, self.model_name + '_metrics.csv')
        mdf.to_csv(mdf_file)
        if meta_data is None:
            meta_data = {}
        if multiplier != 1:
            log_epochs = []
            for epoch in self.log_epochs:
                e = {}
                for k, v in epoch.items():
                    e[k] = v * multiplier
                log_epochs.append(e)
        else:
            log_epochs = self.log_epochs
        result = {'epochs': log_epochs, 'model_name': self.model_name} | meta_data
        json_file = os.path.join(result_path, self.model_name + '_metrics.json')
        with open(json_file, 'w', encoding='utf-8') as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2, sort_keys=False)
        send_metrics(task, self.model_name, [mdf_file, json_file])


# noinspection DuplicatedCode
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
