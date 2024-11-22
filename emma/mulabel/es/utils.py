import logging
from datetime import timedelta, datetime
from typing import Callable, Dict, Any, List, Optional

import numpy as np
from elasticsearch import Elasticsearch
from sklearn.metrics import roc_curve
from tqdm import tqdm

logger = logging.getLogger('mulabel.es.utils')


def _get_date_ranges(start_date: datetime, end_date: datetime, interval_hours: int = 24):
    """Generate date ranges for chunked processing"""
    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + timedelta(hours=interval_hours), end_date)
        yield current_date, next_date
        current_date = next_date


def load_data(client: Elasticsearch, collection: str, start_date: datetime, end_date: datetime,
              callback: Callable[[Dict[str, Any]], bool]) -> None:
    # Process data in date chunks
    date_ranges = list(_get_date_ranges(start_date, end_date))

    for start, end in tqdm(date_ranges, desc='Processing date ranges'):
        # Get documents for the current date range
        query = {
            'size': 10000,
            'query': {
                'range': {
                    'date': {
                        'gte': start.astimezone().isoformat(),
                        'lt': end.astimezone().isoformat()
                    }
                }
            },
            '_source': ['a_uuid', 'a_id', 'date', 'text', 'm_bge_m3', 'label'],
            'sort': {
                'date': 'asc'
            }
        }

        response = client.search(
            index=collection,
            body=query,
        )
        # logger.info(
        #     'Response for range [%s][%s]', start.astimezone().isoformat(), end.astimezone().isoformat()
        # )
        hits = response['hits']['hits']
        for doc in hits:
            if not callback(doc['_source']):
                break


def find_similar_old(client: Elasticsearch, collection: str, uuid: str, vector: List[float],
                     item_callback: Callable[[Dict[str, Any], float], bool],
                     size: int = 50, passage_targets: List[int] = None,
                     passage_cat: int = -1) -> int:
    query = {
        'size': size,
        'query': {
            'bool': {
                'must_not': [
                    {
                        'term': {
                            'a_uuid': uuid
                        }
                    }
                ],
                'must': [{
                    'knn': {
                        'field': 'm_bge_m3',
                        'query_vector': vector,
                        'k': size
                    }
                }]
            }
        },
        '_source': ['a_uuid', 'a_id', 'date', 'text', 'm_bge_m3', 'label', 'label_info'],
        'sort': {
            '_score': 'desc'
        }
    }
    if passage_targets is not None and len(passage_targets) > 0:
        if 'filter' not in query['query']['bool']:
            query['query']['bool']['filter'] = []

        query['query']['bool']['filter'].append(
            {'terms': {'passage_targets': passage_targets}}
        )
    if passage_cat >= 0:
        if 'filter' not in query['query']['bool']:
            query['query']['bool']['filter'] = []

        query['query']['bool']['filter'].append(
            {'term': {'passage_cat': passage_cat}}
        )

    response = client.search(index=collection, body=query)
    for doc in response['hits']['hits']:
        if not item_callback(doc['_source'], doc['_score']):
            break
    return len(response['hits']['hits'])


class LabelStats:

    @staticmethod
    def find_optimal_threshold(y_true, y_prob):
        if len(y_true) == 0:
            return 0.0
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_j = tpr - fpr  # Youden's J statistic
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold

    def __init__(self, label):
        self.label = label
        self.opt = 0.0
        self.min = 0.0
        self.max = 0.0
        self.mean = 0.0
        self.pos = 0.0
        self.neg = 0.0
        self.num = 0.0

    def compute(self, y_true, y_prob):
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        tmp = y_prob * y_true
        tmp = tmp[tmp > 0]
        self.opt = LabelStats.find_optimal_threshold(y_true, y_prob)
        self.min = np.min(tmp) if len(tmp) > 0 else 0.0
        self.max = np.max(tmp) if len(tmp) > 0 else 0.0
        self.mean = np.mean(tmp) if len(tmp) > 0 else 0.0
        self.pos = np.sum(y_true)
        self.neg = y_true.shape[0] - np.sum(y_true)
        self.num = y_true.shape[0]


class SimilarParams:
    def __init__(self, collection: str, not_uuid: str, vector: List[float],
                 passage_targets: List[int] = None,
                 passage_cat: int = -1,
                 size: int = 10):
        self.collection = collection
        self.size = size
        self.uuid = not_uuid
        self.vector = vector
        self.passage_targets = passage_targets
        self.passage_cat = passage_cat


class State:
    def __init__(self, item: Dict[str, Any], text_field: Optional[str] = None):
        self.item: Dict[str, Any] = item
        self.hit: Dict[str, Any] = {}
        self.hits: List[Dict[str, Any]] = []
        self.similar_texts = []
        self.similar_scores = []
        self.index: int = 0
        self.total: int = 0
        self.text_field: Optional[str] = text_field
        self.text: Optional[str] = item[text_field] if text_field is not None else None
        self.data: Dict[str, Any] = {}
        self.score: float = 0.0

    def set_text(self, text: str):
        self.text = text

    def on_hit(self, hit: Dict[str, Any], score: float, index, total):
        self.hit = hit
        self.score = score
        self.total = total
        self.index = index
        self.hits.append(hit)
        self.similar_scores.append(score)
        if self.text_field is not None:
            self.similar_texts.append(hit[self.text_field])

    def pop(self):
        self.hits.pop()
        self.similar_scores.pop()
        if self.text_field is not None:
            self.similar_texts.pop()


def find_similar(client: Elasticsearch, params: SimilarParams,
                  state: State, item_callback: Callable[[State], bool]) -> int:
    query = {
        'size': params.size,
        'query': {
            'bool': {
                'must_not': [
                    {
                        'term': {
                            'a_uuid': params.uuid
                        }
                    }
                ],
                'must': [{
                    'knn': {
                        'field': 'm_bge_m3',
                        'query_vector': params.vector,
                        'k': params.size * 2,
                        'num_candidates': params.size * 100
                    }
                }]
            }
        },
        '_source': ['a_uuid', 'a_id', 'date', 'text', 'm_bge_m3', 'label', 'label_info'],
        'sort': {
            '_score': 'desc'
        }
    }

    if params.passage_targets is not None and len(params.passage_targets) > 0:
        if 'filter' not in query['query']['bool']:
            query['query']['bool']['filter'] = []
        query['query']['bool']['filter'].append(
            {'terms': {'passage_targets': params.passage_targets}}
        )

    if params.passage_cat >= 0:
        if 'filter' not in query['query']['bool']:
            query['query']['bool']['filter'] = []
        query['query']['bool']['filter'].append(
            {'term': {'passage_cat': params.passage_cat}}
        )

    response = client.search(index=params.collection, body=query)
    total = len(response['hits']['hits'])
    for i, doc in enumerate(response['hits']['hits']):
        state.on_hit(doc['_source'], doc['_score'], i, total)
        if not item_callback(state):
            break
    return total


class LabelDecider:

    def __init__(self, labels, calibration):
        self.labels = labels
        self.calibration = calibration
        self.calibrated = {}
        self.missing = {}
        for label in self.labels:
            if label in calibration:
                self.calibrated[label] = calibration[label]
            else:
                self.missing[label] = {}

    def missing(self, label) -> bool:
        return label in self.missing

    def skip(self, label: Optional[str] = None) -> bool:
        if label is None:
            return len(self.calibrated) <= 0
        return label not in self.calibrated

    def get_score(self, pred_label: str, default: float) -> float:
        if pred_label in self.calibration:
            pass  # predicted was calibrated we return default
        else:
            pass  # skipped totally by calibration we should ignore

