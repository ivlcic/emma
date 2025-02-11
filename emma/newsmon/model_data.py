import os
from typing import Union

import numpy as np
import pandas as pd
import torch
import faiss

from .embd_model import EmbeddingModelWrapper
from ..core.metrics import Metrics


class ModelTestData:
    """
    Loads pre-prepared embeddings in np.ndarray format from disk to this object properties
    The files to load with this class are generated via commands:

    ./mulabel fa init_embed -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte,n_bge_m3
    ./mulabel fa init_embed -c eurlex --ptm_models bge_m3,jinav3,gte,e_bge_m3
    """
    def __init__(self, name: str, data_path: str, embedder: EmbeddingModelWrapper,
                 index_metric: int, metrics: Metrics):
        self.name = name
        self.data_path = data_path
        self.has_v = False
        self.y_true = []
        self.y_pred = []
        self.train_data = {
            'x': np.array([]), 'y_true': np.array([]), 'path': self.data_path + '.' + self.name + '_train.npz',
            'dim': 0,  'count': 0
        }
        self.test_data = {
            'x': np.array([]), 'y_true': np.array([]), 'path': self.data_path + '.' + self.name + '_test.npz',
            'dim': 0, 'count': 0
        }
        self.dev_data = {
            'x': np.array([]), 'y_true': np.array([]), 'path': self.data_path + '.' + self.name + '_dev.npz',
            'dim': 0, 'count': 0
        }
        self.v_data = {
            'x': np.array([]), 'y_id': np.array([]), 'path': self.data_path + '.' + self.name + '_v.npz',
            'dim': 0, 'count': 0
        }
        self.embedder = embedder

        train_data = np.load(self.train_data['path'])  # knowledge input train embeddings matrix
        test_data = np.load(self.test_data['path'])  # test embeddings matrix
        dev_data = np.load(self.dev_data['path'])  # test embeddings matrix
        if os.path.exists(self.v_data['path']):
            v_data = np.load(self.v_data['path'])  # knowledge label descr embeddings matrix
            self.has_v = True
            self.v_data['x'] = v_data['x']
            self.v_data['y_id'] = v_data['y_id']

        self.train_data['x'] = train_data['x']
        self.train_data['y_true'] = train_data['y_true']
        self.train_data['dim'] = np.shape(train_data['x'])[1]
        self.train_data['count'] = np.shape(train_data['y_true'])[0]

        if self.has_v:
            self.embeddings = np.vstack([train_data['x'], self.v_data['x']])  # complete knowledge embeddings matrix
        else:
            self.embeddings = train_data['x']

        self.test_data['x'] = test_data['x']
        self.test_data['y_true'] = test_data['y_true']
        self.test_data['dim'] = np.shape(test_data['x'])[1]
        self.test_data['count'] = np.shape(test_data['y_true'])[0]

        self.dev_data['x'] = dev_data['x']
        self.dev_data['y_true'] = dev_data['y_true']
        self.dev_data['dim'] = np.shape(dev_data['x'])[1]
        self.dev_data['count'] = np.shape(dev_data['y_true'])[0]

        self.metrics = metrics

        # init faiss index from embeddings
        k_dim = np.shape(self.embeddings)[0]  # num samples
        dim = np.shape(self.embeddings)[1]  # sample embedding dim
        index = faiss.IndexHNSWFlat(dim, 64, index_metric)
        index.hnsw.efConstruction = 500  # Controls index construction accuracy/speed trade-off
        index.hnsw.efSearch = 300  # Controls search accuracy/speed trade-off
        # noinspection PyArgumentList
        index.add(self.embeddings)
        self.index = index
        self.k_dim = k_dim
        self.tk_dim = k_dim
        self.dim = dim
        self.temperature = 0.04
        self.top_k = 100
        self.values = torch.tensor([], dtype=torch.float32)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mlknn = None

    def set_hyper(self, temperature: float, top_k: int, lamb: float):
        self.temperature = temperature
        self.top_k = top_k
        if self.has_v:
            k_v = np.vstack([self.train_data['y_true'] * lamb, self.v_data['y_id'] * (1 - lamb)])
        else:
            k_v = self.train_data['y_true'] * lamb
        self.values = torch.from_numpy(k_v.astype(np.float32)).to(self.device)


class ModelTestObjective:

    def _get_trial_params(self):
        return []

    def __init__(self, args, m_data: ModelTestData):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.m_data = m_data
        self.args = args

        self.csv_file_name = os.path.join(args.data_result_dir, f'{m_data.metrics.model_name}_trials.csv')
        columns = ['Trial', 'Time', 'Value']
        columns += [f'param_{key}' for key in self._get_trial_params()]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.csv_file_name, index=False)

    def log_to_csv(self, trial_number, t, acc, params):
        row = {'Trial': trial_number, 'Time': t, 'Value': acc}
        row.update({f'param_{key}': val for key, val in params.items()})
        df = pd.DataFrame([row])
        df.to_csv(self.csv_file_name, mode="a", header=False, index=False, encoding="utf-8")
