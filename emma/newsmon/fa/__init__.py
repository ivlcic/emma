import json
import logging
import os
import random
import time
from argparse import ArgumentParser
from typing import Dict, Any, List

import faiss
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import optim, nn
from tqdm import tqdm

from ..const import __supported_languages, __label_split_names
from ..embd_model import EmbeddingModelWrapperFactory, EmbeddingModelWrapper
from ..mlknn import MLkNN
from ..model import RaeExt
from ..model_data import ModelTestData, ModelTestObjective
from ..utils import compute_arg_collection_name, get_index_path, load_data, chunk_data, init_labeler, filter_metrics
from ...core.args import CommonArguments
from ...core.labels import Labeler
from ...core.metrics import Metrics
from ...core.models import retrieve_model_name_map

logger = logging.getLogger('mulabel.fa')


# noinspection DuplicatedCode
def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    CommonArguments.result_dir(module_name, parser, ('-r', '--data_result_dir'))
    parser.add_argument(
        '-c', '--collection', help='Collection to manage.', type=str, default='mulabel'
    )
    parser.add_argument(
        '-l', '--lang',
        help=f'Use only languages (filter everything else out). '
             f'You can use a comma separated list of {__supported_languages}',
        type=str,
    )
    parser.add_argument(
        '--public', help='Use only publicly available data.',
        action='store_true', default=False
    )
    parser.add_argument(
        '--seed_only', help='Use only seed labels (not all article labels).',
        action='store_true', default=False
    )
    parser.add_argument(
        '--suffix', help='Use suffix when processing files.',
        type=str
    )
    parser.add_argument(
        '--test_l_class', type=str, help=f'Test specified label class.',
        choices=['all'].extend(__label_split_names), default='all'
    )
    parser.add_argument(
        '--ptm_models',
        help=f'Use only ptm_models (filter everything else out). '
             f'You can use a comma separated list of {retrieve_model_name_map.keys()}',
        type=str,
        default='bge_m3'
    )


# noinspection DuplicatedCode
def fa_init_embed(args) -> int:
    """
    ./newsmon fa init_embed -c newsmon -l sl --public --ptm_models bge_m3,jinav3,gte,n_bge_m3
    ./newsmon fa init_embed -c eurlex --ptm_models bge_m3,jinav3,gte,e_bge_m3
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)
    train_data_as_dicts, _ = load_data(args, args.collection + '_train')  # we load the train data
    dev_data_as_dicts, _ = load_data(args, args.collection + '_dev')  # we load the validation data
    test_data_as_dicts, _ = load_data(args, args.collection + '_test')  # we load the test data
    data = {
        'train': train_data_as_dicts,
        'dev': dev_data_as_dicts,
        'test': test_data_as_dicts
    }
    inputs: Dict[str: Dict[str, Any]] = {}
    for model_name in models:
        inputs[model_name] = {
            'train': {'samples': 0, 'y_true': [], 'x': []},
            'dev': {'samples': 0, 'y_true': [], 'x': []},
            'test': {'samples': 0, 'y_true': [], 'x': []},
        }

    total = 0
    for k, data_as_dicts in data.items():
        for chunk in chunk_data(data_as_dicts, chunk_size=384):
            logger.info(f'Processing {k} at {total/len(data_as_dicts) * 100:.2f}%')
            total += len(chunk)
            texts: List[str] = [d['text'] for d in chunk]
            label_v = np.array([labeler.vectorize([d['label']]) for d in chunk])
            label_v = np.squeeze(label_v, axis=1)
            for model_name, model in models.items():
                ret = model.embed(texts)
                batch_size = ret.shape[0]
                inputs[model_name][k]['samples'] += batch_size
                inputs[model_name][k]['x'].append(ret)
                inputs[model_name][k]['y_true'].append(label_v)

    index_path = get_index_path(args)
    for model_name in models:
        for k, data_as_dicts in data.items():
            data_dict = inputs[model_name][k]
            data_dict['x'] = np.vstack(inputs[model_name][k]['x'])
            data_dict['y_true'] = np.vstack(inputs[model_name][k]['y_true'])
            # noinspection PyTypeChecker
            np.savez_compressed(index_path + f'.{model_name}_{k}.npz', **data_dict)

    return 0


def init_model_data(args,
                    labeler: Labeler,
                    index_metric: int,
                    prefix: str,
                    models: Dict[str, EmbeddingModelWrapper]) -> Dict[str, ModelTestData]:
    index_path = get_index_path(args)
    # static model data
    model_data = {}
    for m_name, model in models.items():
        t0 = time.time()
        suffix = ''
        if args.test_l_class != 'all':
            suffix = '_' + args.test_l_class
        m = Metrics(f'{prefix}_{m_name}_{args.collection}{suffix}', labeler.get_type_code())
        model_data[m_name] = ModelTestData(m_name, str(index_path), model, index_metric, m)
        logger.info(f'Loaded {m_name} data in {(time.time() - t0):8.2f} seconds')
    return model_data


# noinspection DuplicatedCode
def fa_mine_hard_neg(args) -> int:
    """
    ./newsmon fa mine_hard_neg -c newsmon -l sl
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)

    train_coll_name = args.collection + '_train'
    test_coll_name = args.collection + '_test'
    data_as_dicts, _ = load_data(args, train_coll_name)

    model_data = init_model_data(args, labeler, faiss.METRIC_INNER_PRODUCT, 'ignore', models)
    batch_size = 64
    k = 500
    nn = 15
    num_samples = len(data_as_dicts)
    model_samples = {}
    for m_name, m_item in model_data.items():
        model_samples[m_name] = []

    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Hard Negative Sampling"):
        end_idx = min(start_idx + batch_size, num_samples)
        indices = list(range(start_idx, end_idx))
        texts = [item['text'] for item in data_as_dicts[start_idx:end_idx]]
        labels = [item['label'] for item in data_as_dicts[start_idx:end_idx]]
        for m_name, m_data in model_data.items():
            y_true = m_data.train_data['y_true']
            yb_true = y_true[start_idx:end_idx]  # current batch
            query_vectors = m_data.train_data['x'][start_idx:end_idx]
            query_vectors = query_vectors.astype(np.float32)  # Ensure correct dtype

            # noinspection PyArgumentList
            batch_sim, batch_indices = m_data.index.search(query_vectors, k + 1)

            for i, sim_sample_indices in enumerate(batch_indices):
                sample = {
                    'idx': indices[i], 'query': texts[i], 'label': labels[i],
                    'pos': [], 'pos_l': [], 'pos_i': [],
                    'neg': [], 'neg_l': [], 'neg_i': [], 'inb_neg_i': [],
                }
                for idx in sim_sample_indices[1:]:  # Skip self-match (first result)
                    y_sim_sample = y_true[idx]
                    if len(sample['neg']) >= nn and len(sample['pos']) >= 1:
                        break
                    if not np.any(yb_true[i] & y_sim_sample):
                        if len(sample['neg']) < nn:
                            sample['neg'].append(data_as_dicts[idx]['text'])
                            sample['neg_l'].append(data_as_dicts[idx]['label'])
                            sample['neg_i'].append(int(idx))
                    elif len(sample['pos']) < 1:
                        sample['pos'].append(data_as_dicts[idx]['text'])
                        sample['pos_l'].append(data_as_dicts[idx]['label'])
                        sample['pos_i'].append(int(idx))

                num_missing = nn - len(sample['neg'])
                if num_missing > 0:
                    in_batch_neg = [num for num in range(start_idx, end_idx) if num not in sim_sample_indices]
                    if len(in_batch_neg) <= num_missing:
                        logger.error(f'Missing {num_missing} greater than '
                                     f'available samples [{in_batch_neg}] in {indices[i]} ')
                        # use prev batch
                        in_batch_neg = [num for num in range(start_idx - batch_size, end_idx - batch_size)
                                        if num not in sim_sample_indices]
                        if len(in_batch_neg) <= num_missing:
                            logger.error(f'Missing {num_missing} greater than '
                                         f'available samples [{in_batch_neg}] in {indices[i]} ')
                            continue
                    missing = random.sample(in_batch_neg, num_missing)
                    for idx in missing:
                        sample['neg'].append(data_as_dicts[idx]['text'])
                        sample['neg_l'].append(data_as_dicts[idx]['label'])
                        sample['neg_i'].append(int(idx))
                        sample['inb_neg_i'].append(int(idx))

                if len(sample['pos']) == 0:
                    logger.warning(f'Missing positive sample for {indices[i]}')
                else:
                    model_samples[m_name].append(sample)

        # if start_idx >= 128:  # for debug
        #    break

    for m_name, m_item in model_samples.items():
        jsonl_file_name = f'{m_name}_{args.collection}_hn.jsonl'
        jsonl_path = os.path.join(args.data_in_dir, jsonl_file_name)
        with open(jsonl_path, 'w', encoding='utf-8') as fp:
            for i, sample in enumerate(model_samples[m_name]):
                # noinspection PyTypeChecker
                json.dump(sample, fp)
                fp.write('\n')
    return 0


__best_hyper_params = {
    'newsmon': {
        'raexmc': {
            'bge_m3': {
                'top_k': 13, 'lambda': 0.999, 'temperature': 0.091
            },
            'n_bge_m3': {
                'top_k': 13, 'lambda': 0.815, 'temperature': 0.098
            }
        },
        'raexmcsim': {
            'bge_m3': {
                'top_k': 16, 'lambda': 0.986, 'temperature': 0.067
            },
            'n_bge_m3': {
                'top_k': 12, 'lambda': 0.807, 'temperature': 0.066
            }
        },
    },
    'eurlex': {
        'raexmc': {
            'bge_m3': {
                'top_k': 50, 'lambda': 1.0, 'temperature': 0.04
            },
            'e_bge_m3': {
                'top_k': 50, 'lambda': 1.0, 'temperature': 0.04
            }
        },
        'raexmcsim': {
            'bge_m3': {
                'top_k': 10, 'lambda': 0.999, 'temperature': 0.010
            },
            'e_bge_m3': {
                'top_k': 52, 'lambda': 0.999, 'temperature': 0.017
            }
        }
    }
}


# noinspection DuplicatedCode
def fa_test_rae(args) -> int:
    """
    ./newsmon fa test_rae -c newsmon -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)

    method = 'raexmc'
    model_data = init_model_data(args, labeler, faiss.METRIC_L2, method, models)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # subject to grid search
    # lamb = 0.5
    batch_size = 384
    lamb = 1
    topk = 50  # Number of nearest neighbors to retrieve
    threshold = 0.5  # Probability to classify as a positive
    temper = 0.04
    corpus = args.collection_conf
    for m_name, m_data in model_data.items():
        if (corpus in __best_hyper_params and method in __best_hyper_params[corpus]
                and m_name in __best_hyper_params[corpus][method]):
            params = __best_hyper_params[corpus][method][m_name]
            logger.info(f'Processing RAE-XMC distance metrics for {m_name} with optimal hyper-parameters {params}')
            m_data.set_hyper(params['temperature'], params['top_k'], params['lambda'])
        else:
            m_data.set_hyper(temper, topk, lamb)

    t1 = time.time()
    for model_name, m_data in model_data.items():
        logger.info(f'Processing RAE-XMC distance eval for {model_name}')
        for start_idx in tqdm(range(0, m_data.test_data['count'], batch_size),
                              desc='Processing RAE-XMC distance eval.'):
            end_idx = min(start_idx + batch_size, m_data.test_data['count'])
            query_vectors = m_data.test_data['x'][start_idx:end_idx]
            yl_true = m_data.test_data['y_true'][start_idx:end_idx]

            # Search for the topk nearest neighbors for all query vectors in the batch
            # noinspection PyArgumentList
            sim, indices = m_data.index.search(query_vectors, m_data.top_k)  # Batched search
            sim = torch.from_numpy(sim).to(device)
            # Convert similarities to probability distribution using softmax
            sim = F.softmax(-torch.sqrt(sim) / m_data.temperature, dim=-1)
            # Initialize qKT tensor for the batch
            qkt = torch.zeros((end_idx - start_idx, m_data.k_dim), dtype=torch.float32).to(device)
            # Assign values to specific indices for each item in the batch
            for i in range(end_idx - start_idx):
                qkt[i, indices[i]] = sim[i]

            # Compute probabilities using matrix multiplication
            probabilities = torch.matmul(qkt, m_data.values)
            yl_prob = (probabilities > threshold).cpu().numpy()
            m_data.y_pred.extend(yl_prob)
            m_data.y_true.extend(yl_true)

    logger.info(f'Measured performance in {(time.time() - t1):8.2f} seconds')
    logger.info(f'Computing metrics')
    for model_name, m_data in model_data.items():
        logger.info(f'Processing RAE-XMC metrics for {model_name}')
        y_true_m, y_pred_m = filter_metrics(args, labeler, m_data.y_true, m_data.y_pred)
        m_data.metrics(y_true_m, y_pred_m, 'test/', threshold)
        meta = {'num_samples': np.shape(y_true_m)[0], 'num_labels': np.shape(y_true_m)[1]}
        m_data.metrics.dump(args.data_result_dir, meta, None, 100)
    logger.info(f'Computation done in {(time.time() - t1):8.2f} seconds')
    return 0


# noinspection DuplicatedCode
def fa_test_zshot(args) -> int:
    """
    ./newsmon fa test_zshot -c newsmon -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)

    model_data = init_model_data(args, labeler, faiss.METRIC_L2, 'zshot', models)

    batch_size = 384
    t0 = time.time()
    for model_name, m_data in model_data.items():
        logger.info(f'Processing zero-shot eval for {model_name}')
        for start_idx in tqdm(range(0, m_data.test_data['count'], batch_size), desc='Processing zero-shot eval.'):
            end_idx = min(start_idx + batch_size, m_data.test_data['count'])
            query_vectors = m_data.test_data['x'][start_idx:end_idx]
            yl_true = m_data.test_data['y_true'][start_idx:end_idx]

            # Search for the topk nearest neighbors for all query vectors in the batch
            # noinspection PyArgumentList
            sim, indices = m_data.index.search(query_vectors, 1)  # Batched search
            yl_pred = m_data.train_data['y_true'][indices].squeeze()
            m_data.y_pred.extend(yl_pred)
            m_data.y_true.extend(yl_true)

    logger.info(f'Measured performance in {(time.time() - t0):8.2f} seconds')
    logger.info(f'Computing metrics')
    for model_name, m_data in model_data.items():
        logger.info(f'Processing zero-shot metrics for {model_name}')
        y_true_m, y_pred_m = filter_metrics(args, labeler, m_data.y_true, m_data.y_pred)
        m_data.metrics(y_true_m, y_pred_m, 'test/')
        meta = {'num_samples': np.shape(y_true_m)[0], 'num_labels': np.shape(y_true_m)[1]}
        m_data.metrics.dump(args.data_result_dir, meta, None, 100)

    logger.info(f'Computation done in {(time.time() - t0):8.2f} seconds')
    return 0


# noinspection DuplicatedCode
def fa_test_mlknn(args) -> int:
    """
    ./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)

    model_data = init_model_data(args, labeler, faiss.METRIC_L2, 'mlknn', models)

    def knn_search(mn: str, queries: np.ndarray, k: int):
        # Ensure the queries are 2D
        queries = queries.reshape(-1, queries.shape[1]).astype(np.float32)

        # Perform a batched search for all query vectors
        index = model_data[mn].index
        # noinspection PyArgumentList
        sim, indices = index.search(queries, k=(k + 1))
        indices = np.delete(indices, 0, axis=1)  # remove first - self match

        return indices

    threshold = 0.5
    topk = 50  # Number of nearest neighbors to retrieve
    for m_name, m_data in model_data.items():
        embeddings = m_data.embeddings
        y_true = m_data.train_data['y_true']
        mlknn = MLkNN(m_name, topk, 1, knn_search)
        mlknn.fit(embeddings, y_true)
        m_data.mlknn = mlknn

    batch_size = 384
    t0 = time.time()
    for model_name, m_data in model_data.items():
        logger.info(f'Processing ML-KNN eval for {model_name}')
        for start_idx in tqdm(range(0, m_data.test_data['count'], batch_size), desc='Processing ML-KNN eval.'):
            end_idx = min(start_idx + batch_size, m_data.test_data['count'])
            query_vectors = m_data.test_data['x'][start_idx:end_idx]
            yl_true = m_data.test_data['y_true'][start_idx:end_idx]

            predictions, probabilities = m_data.mlknn.predict(query_vectors)
            m_data.y_pred.extend(probabilities)
            m_data.y_true.extend(yl_true)

    logger.info(f'Measured performance in {(time.time() - t0):8.2f} seconds')
    logger.info(f'Computing metrics')
    for model_name, m_data in model_data.items():
        logger.info(f'Processing ML-KNN metrics for {model_name}')
        y_true_m, y_pred_m = filter_metrics(args, labeler, m_data.y_true, m_data.y_pred)
        m_data.metrics(y_true_m, y_pred_m, 'test/', threshold)
        meta = {'num_samples': np.shape(y_true_m)[0], 'num_labels': np.shape(y_true_m)[1]}
        m_data.metrics.dump(args.data_result_dir, meta, None, 100)

    logger.info(f'Computation done in {(time.time() - t0):8.2f} seconds')
    return 0


# noinspection DuplicatedCode
def fa_test_rae_sim(args) -> int:
    """
    ./newsmon fa test_rae_sim -c newsmon -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)

    method = 'raexmcsim'
    model_data = init_model_data(args, labeler, faiss.METRIC_INNER_PRODUCT, method, models)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # subject to grid search
    # lamb = 0.5
    batch_size = 384
    lamb = 1
    topk = 50  # Number of nearest neighbors to retrieve
    threshold = 0.5  # Probability to classify as a positive
    temper = 0.04
    corpus = args.collection_conf
    for m_name, m_data in model_data.items():
        if (corpus in __best_hyper_params and method in __best_hyper_params[corpus]
                and m_name in __best_hyper_params[corpus][method]):
            params = __best_hyper_params[corpus][method][m_name]
            logger.info(f'Processing RAE-XMC similarity metrics for {m_name} with optimal hyper-parameters {params}')
            m_data.set_hyper(params['temperature'], params['top_k'], params['lambda'])
        else:
            m_data.set_hyper(temper, topk, lamb)

    t1 = time.time()
    for model_name, m_data in model_data.items():
        logger.info(f'Processing RAE-XMC similarity eval for {model_name}')
        for start_idx in tqdm(range(0, m_data.test_data['count'], batch_size),
                              desc='Processing RAE-XMC similarity (dot-product) eval.'):
            end_idx = min(start_idx + batch_size, m_data.test_data['count'])
            query_vectors = m_data.test_data['x'][start_idx:end_idx]
            yl_true = m_data.test_data['y_true'][start_idx:end_idx]

            # Search for the topk nearest neighbors for all query vectors in the batch
            # noinspection PyArgumentList
            sim, indices = m_data.index.search(query_vectors, m_data.top_k)  # Batched search
            sim = torch.from_numpy(sim).to(device)
            # Convert similarities to probability distribution using softmax
            sim = F.softmax(sim / m_data.temperature, dim=-1)
            # Initialize qKT tensor for the batch
            qkt = torch.zeros((end_idx - start_idx, m_data.k_dim), dtype=torch.float32).to(device)
            # Assign values to specific indices for each item in the batch
            for i in range(end_idx - start_idx):
                qkt[i, indices[i]] = sim[i]

            # Compute probabilities using matrix multiplication
            probabilities = torch.matmul(qkt, m_data.values)
            yl_prob = (probabilities > threshold).cpu().numpy()
            m_data.y_pred.extend(yl_prob)
            m_data.y_true.extend(yl_true)

    logger.info(f'Measured performance in {(time.time() - t1):8.2f} seconds')
    logger.info(f'Computing metrics')
    for model_name, m_data in model_data.items():
        logger.info(f'Processing RAE-XMC metrics for {model_name}')
        y_true_m, y_pred_m = filter_metrics(args, labeler, m_data.y_true, m_data.y_pred)
        m_data.metrics(y_true_m, y_pred_m, 'test/', threshold)
        meta = {'num_samples': np.shape(y_true_m)[0], 'num_labels': np.shape(y_true_m)[1]}
        m_data.metrics.dump(args.data_result_dir, meta, None, 100)
    logger.info(f'Computation done in {(time.time() - t1):8.2f} seconds')
    return 0


class RaeObjective(ModelTestObjective):

    def _get_trial_params(self):
        return ['temperature', 'topk', 'lambda']

    def __init__(self, args, m_data: ModelTestData, labeler: Labeler, dist_metric: bool = False,
                 on_test_set: bool = False, batch_size: int = 384):
        super().__init__(args, m_data, '_test' if on_test_set else '_dev')
        self.labeler = labeler
        self.batch_size = batch_size
        self.dist_metric = dist_metric
        self.on_test_set = on_test_set

    def __call__(self, trial):
        threshold = 0.5
        temper = trial.suggest_float('temperature', 0.01, 0.1)
        top_k = trial.suggest_int('top_k', 10, 100)
        lamb = trial.suggest_float('lambda', 0.1, 1.0)

        self.m_data.set_hyper(temper, top_k, lamb)
        t0 = time.time()
        data_set = self.m_data.dev_data
        if self.on_test_set:
            data_set = self.m_data.test_data
        data_len = data_set['count']
        for start_idx in tqdm(range(0, data_len, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, data_len)
            query_vectors = data_set['x'][start_idx:end_idx]
            yl_true = data_set['y_true'][start_idx:end_idx]

            # Search for the topk nearest neighbors for all query vectors in the batch
            # noinspection PyArgumentList
            sim, indices = self.m_data.index.search(query_vectors, self.m_data.top_k)  # Batched search
            sim = torch.from_numpy(sim).to(self.device)
            # Convert similarities to probability distribution using softmax
            if self.dist_metric:
                sim = F.softmax(-torch.sqrt(sim) / self.m_data.temperature, dim=-1)
            else:
                sim = F.softmax(sim / self.m_data.temperature, dim=-1)
            # Initialize qKT tensor for the batch
            qkt = torch.zeros((end_idx - start_idx, self.m_data.k_dim), dtype=torch.float32).to(self.device)
            # Assign values to specific indices for each item in the batch
            for i in range(end_idx - start_idx):
                qkt[i, indices[i]] = sim[i]

            # Compute probabilities using matrix multiplication
            probabilities = torch.matmul(qkt, self.m_data.values)
            yl_prob = (probabilities > threshold).cpu().numpy()
            self.m_data.y_pred.extend(yl_prob)
            self.m_data.y_true.extend(yl_true)

        t1 = (time.time() - t0)
        y_true_m, y_pred_m = filter_metrics(self.args, self.labeler, self.m_data.y_true, self.m_data.y_pred)
        result = accuracy_score(y_true_m, y_pred_m) * 100
        self.log_to_csv(trial.number, t1, result, trial.params)
        self.m_data.y_true = []
        self.m_data.y_pred = []
        return result


# noinspection DuplicatedCode
def fa_optuna_rae_sim(args) -> int:
    """
    ./newsmon fa optuna_rae_sim -c newsmon -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)
    model_data = init_model_data(args, labeler, faiss.METRIC_INNER_PRODUCT, 'raexmcsim', models)

    for m_name, m_data in model_data.items():
        logger.info(f'Processing RAE-XMC similarity based eval for {m_data.name}')
        objective = RaeObjective(args, m_data, labeler)
        study = optuna.create_study(direction="maximize")  # Assuming higher metric value is better
        study.optimize(objective, n_trials=1000)  # Adjust n_trials as needed
        objective.log_to_csv(study.best_trial.number, 0, study.best_trial.values[0], study.best_trial.params)
    return 0


# noinspection DuplicatedCode
def fa_optuna_rae(args) -> int:
    """
    ./newsmon fa optuna_rae -c newsmon -l sl --public --ptm_models bge_m3
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)
    model_data = init_model_data(args, labeler, faiss.METRIC_L2, 'raexmc', models)

    for m_name, m_data in model_data.items():
        logger.info(f'Processing RAE-XMC distance based eval for {m_data.name}')
        objective = RaeObjective(args, m_data, labeler, True)
        study = optuna.create_study(direction="maximize")  # Assuming higher metric value is better
        study.optimize(objective, n_trials=1000)  # Adjust n_trials as needed
        objective.log_to_csv(study.best_trial.number, 0, study.best_trial.values[0], study.best_trial.params)
    return 0


def fa_train_rae_sim_ext(args) -> int:
    """
    ./newsmon fa train_rae_sim_ext -c newsmon -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)
    method = 'raexmcsim'
    model_data = init_model_data(args, labeler, faiss.METRIC_INNER_PRODUCT, method, models)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 384
    num_epochs = 10

    lamb = 1
    topk = 50  # Number of nearest neighbors to retrieve
    temper = 0.04

    corpus = args.collection_conf
    for m_name, m_data in model_data.items():
        if (corpus in __best_hyper_params and method in __best_hyper_params[corpus]
                and m_name in __best_hyper_params[corpus][method]):
            params = __best_hyper_params[corpus][method][m_name]
            logger.info(f'RAE-XMC similarity model {m_name} with optimal hyper-parameters {params} loaded.')
            m_data.set_hyper(params['temperature'], params['top_k'], 1)  # we will learn lambda matrix
        else:
            logger.info(f'RAE-XMC similarity model {m_name} with default hyper-parameters loaded.')
            m_data.set_hyper(temper, topk, 1)

    for m_name, m_data in model_data.items():
        logger.info(f'Processing RAE-XMC similarity based train for {m_data.name}')
        data_set = m_data.dev_data
        data_len = data_set['count']

        model = RaeExt(
            values_matrix=m_data.values,
            index=m_data.index,
            top_k=m_data.top_k,
            k_dim=np.shape(m_data.train_data['y_true'])[1],
            temperature=m_data.temperature,
            dist_metric=False,
            device=device
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for start_idx in tqdm(range(0, data_len, batch_size)):
                end_idx = min(start_idx + batch_size, data_len)
                query_vectors = data_set['x'][start_idx:end_idx]
                yl_true = data_set['y_true'][start_idx:end_idx]
                query_vectors, yl_true = query_vectors.to(device), yl_true.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(query_vectors)

                # Compute loss
                loss = criterion(outputs, yl_true)

                # Backward pass and optimization step
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (data_len / batch_size):.4f}')
    return 0