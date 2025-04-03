import logging
import time

from argparse import ArgumentParser
from typing import Dict, Any, List

import numpy as np
import torch
import faiss

import torch.nn.functional as F
from tqdm import tqdm

from ..embd_model import EmbeddingModelWrapperFactory, EmbeddingModelWrapper
from ...core.args import CommonArguments
from ...core.labels import Labeler
from ...core.metrics import Metrics
from ...core.models import retrieve_model_name_map
from ..const import __supported_languages, __label_split_names
from ..utils import compute_arg_collection_name, get_index_path, load_data, chunk_data, init_labeler, filter_metrics
from ..model_data import ModelTestData

logger = logging.getLogger('newsmon.bo')


# noinspection DuplicatedCode
def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    CommonArguments.result_dir(module_name, parser, ('-r', '--data_result_dir'))
    parser.add_argument(
        '-c', '--collection', help='Collection to manage.', type=str, default='newsmon'
    )
    parser.add_argument(
        '-l', '--lang',
        help=f'Use only languages (filter everything else out). '
             f'You can use a comma separated list of {__supported_languages}',
        type=str,
        default='sl'
    )
    parser.add_argument(
        '--public', help='Use only publicly available data.',
        action='store_true', default=True
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
def bo_init_embed(args) -> int:
    """
    ./mulabel bo init_embed -c newsmon -l sl --public --ptm_models bge_m3,jinav3,gte,n_bge_m3
    ./mulabel bo init_embed -c eurlex --ptm_models bge_m3,jinav3,gte,e_bge_m3
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler, _ = init_labeler(args)
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


# noinspection DuplicatedCode
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
def bo_test_rae(args) -> int:
    """
    Warning this uses L2 metric same as original paper authors, although in paper they use similarity
    (dot product of normalized vectors)
    ./newsmon bo test_rae -c newsmon -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler, _ = init_labeler(args)

    model_data = init_model_data(args, labeler, faiss.METRIC_L2, 'raexmc', models)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # subject to grid search
    # lamb = 0.5
    batch_size = 384
    lamb = 1
    topk = 50  # Number of nearest neighbors to retrieve
    threshold = 0.5  # Probability to classify as a positive
    temper = 0.04
    for m_name, m_data in model_data.items():
        m_data.set_hyper(temper, topk, lamb)

    t1 = time.time()
    for model_name, m_data in model_data.items():
        logger.info(f'Processing RAE-XMC eval for {model_name}')
        for start_idx in tqdm(range(0, m_data.test_data['count'], batch_size), desc='Processing RAE-XMC eval.'):
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
def bo_test_rae_sim(args) -> int:
    """
    ./newsmon bo test_rae -c newsmon -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler, _ = init_labeler(args)

    model_data = init_model_data(args, labeler, faiss.METRIC_INNER_PRODUCT, 'raexmcsim', models)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # subject to grid search
    # lamb = 0.5
    batch_size = 384
    lamb = 1
    topk = 50  # Number of nearest neighbors to retrieve
    threshold = 0.5  # Probability to classify as a positive
    temper = 0.04
    for m_name, m_data in model_data.items():
        m_data.set_hyper(temper, topk, lamb)

    t1 = time.time()
    for model_name, m_data in model_data.items():
        logger.info(f'Processing RAE-XMC eval for {model_name}')
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


# noinspection DuplicatedCode
def bo_test_zshot(args) -> int:
    """
    ./newsmon bo test_zshot -c newsmon -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler, _ = init_labeler(args)

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
