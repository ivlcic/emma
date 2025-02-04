import ast
import json
import os
import logging
import random
import time

from argparse import ArgumentParser
from typing import Dict, Any, List, LiteralString

import numpy as np
import pandas as pd
import torch
import faiss

import torch.nn.functional as F
from tqdm import tqdm

from ..embd_model import EmbeddingModelWrapperFactory, EmbeddingModelWrapper
from ..mlknn import MLkNN
from ...core.args import CommonArguments
from ...core.labels import Labeler
from ...core.metrics import Metrics
from ...core.models import retrieve_model_name_map
from ..const import __supported_languages, __label_split_names, __supported_passage_sizes
from ..utils import compute_arg_collection_name, load_data, chunk_data, init_labeler, filter_metrics

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
        '--passage_sizes', help=f'When calibrating use passage_sizes '
                                f'You can use a comma separated list of {__supported_passage_sizes}',
        type=str,
    )
    parser.add_argument(
        '--run_id', type=int, help=f'Run id for marking consecutive runs.', default=0
    )
    parser.add_argument(
        '--lrp_size', type=int, help=f'LRP size.', default=1
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
        default=next(iter(retrieve_model_name_map))
    )


def _get_index_path(args) -> LiteralString | str | bytes:
    index_path = os.path.join(args.data_result_dir, 'index')
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    index_path = os.path.join(args.data_result_dir, 'index', args.collection + '_train')
    return index_path


def fa_init_embed(args) -> int:
    """
    ./mulabel fa init_embed -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte
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

    index_path = _get_index_path(args)
    for model_name in models:
        for k, data_as_dicts in data.items():
            data_dict = inputs[model_name][k]
            data_dict['x'] = np.vstack(inputs[model_name][k]['x'])
            data_dict['y_true'] = np.vstack(inputs[model_name][k]['y_true'])
            # noinspection PyTypeChecker
            np.savez_compressed(index_path + f'.{model_name}_{k}.npz', **data_dict)

    return 0


# noinspection DuplicatedCode
def _select_random_sentences(passages, max_chars):
    selected_sentences = []
    total_chars = 0

    # Shuffle the list to ensure randomness
    random.shuffle(passages)

    for sentence in passages:
        # Check if adding the current sentence exceeds the max character limit
        if total_chars + len(sentence) <= max_chars:
            selected_sentences.append(sentence)
            total_chars += len(sentence)
        else:
            break

    return ' '.join(selected_sentences)


# noinspection DuplicatedCode
def fa_init_rae_v(args) -> int:
    """
    ./mulabel fa init_rae_v -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)

    # read label descriptions / passages
    label_descr_file_path = os.path.join(args.data_in_dir, f'{args.collection}_labels_descr.csv')
    if not os.path.exists(label_descr_file_path):
        logger.warning(f'Missing label description file [{label_descr_file_path}]. '
                       f'Run [./newsmon prep init_pseudo_labels -c mulabel -l sl --public] or similar first!')
        return 1
    label_descr_df = pd.read_csv(label_descr_file_path)
    label_descr_df['passages'] = label_descr_df['passages'].apply(ast.literal_eval)
    label_descr_df['texts'] = label_descr_df['texts'].apply(ast.literal_eval)
    label_descr_df['label_info'] = label_descr_df['label_info'].apply(ast.literal_eval)

    random.seed(2611)
    num_labels = labeler.num_labels
    descr_size_chars = 2000
    labels: Dict[str: Dict[str, Any]] = {}
    for model_name in models:
        labels[model_name] = {}
        labels[model_name]['samples'] = num_labels
        labels[model_name]['v_train'] = []
        labels[model_name]['y_id'] = []

    texts = ['_'] * num_labels
    label_id_map = labeler.labels_to_ids()
    for label in label_descr_df.to_dict('records'):
        if label['passages'] and label['texts']:
            text = _select_random_sentences(label['passages'], descr_size_chars)
        elif label['passages']:
            text = _select_random_sentences(label['passages'], descr_size_chars)
        else:
            random.shuffle(label['texts'])
            text = label['texts'][0]
        l_id = label_id_map[label['label']]
        texts[l_id] = text

    for chunk in chunk_data(texts, chunk_size=384):
        for model_name, model in models.items():
            ret = model.embed(chunk)
            labels[model_name]['v_train'].append(ret)

    index_path = _get_index_path(args)
    for model_name in models:
        data_dict = labels[model_name]
        data_dict['v_train'] = np.vstack(labels[model_name]['v_train'])
        data_dict['y_id'] = np.identity(num_labels)
        # noinspection PyTypeChecker
        np.savez_compressed(index_path + '.' + model_name + '_v.npz', **data_dict)

    return 0


class ModelData:

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
        self.v_data = {
            'x': np.array([]), 'y_id': np.array([]), 'path': self.data_path + '.' + self.name + '_v.npz',
            'dim': 0, 'count': 0
        }
        self.embedder = embedder

        train_data = np.load(self.train_data['path'])  # knowledge input embeddings matrix
        test_data = np.load(self.test_data['path'])  # knowledge input embeddings matrix
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
            self.embeddings = np.vstack([train_data['x'], self.v_data['x']])  # knowledge embeddings matrix
        else:
            self.embeddings = train_data['x']

        self.test_data['x'] = test_data['x']
        self.test_data['y_true'] = test_data['y_true']
        self.test_data['dim'] = np.shape(test_data['x'])[1]
        self.test_data['count'] = np.shape(test_data['y_true'])[0]
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



def init_model_data(args,
                    labeler: Labeler,
                    index_metric: int,
                    prefix: str,
                    models: Dict[str, EmbeddingModelWrapper]) -> Dict[str, ModelData]:
    index_path = _get_index_path(args)
    # static model data
    model_data = {}
    for m_name, model in models.items():
        t0 = time.time()
        suffix = ''
        if args.test_l_class != 'all':
            suffix = '_' + args.test_l_class
        m = Metrics(f'{prefix}_{m_name}_{args.collection}{suffix}', labeler.get_type_code())
        model_data[m_name] = ModelData(m_name, str(index_path), model, index_metric, m)
        logger.info(f'Loaded {m_name} data in {(time.time() - t0):8.2f} seconds')
    return model_data


# noinspection DuplicatedCode
def fa_hard_neg(args) -> int:
    """
    ./mulabel fa hard_neg -c mulabel -l sl
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

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


# noinspection DuplicatedCode
def fa_test_rae(args) -> int:
    """
    ./mulabel fa test_rae -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)

    train_coll_name = args.collection + '_train'
    test_coll_name = args.collection + '_test'
    # data_as_dicts, _ = load_data(args, test_coll_name)

    model_data = init_model_data(args, labeler, faiss.METRIC_L2, 'raexmc_sqrt_', models)

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
        for start_idx in tqdm(range(0, m_data.test_data['count'], batch_size), desc='Processing RAE-XML eval.'):
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
    for m_name, m_data in model_data.items():
        y_true_m, y_pred_m = filter_metrics(args, labeler, m_data.y_true, m_data.y_pred)
        m_data.metrics(y_true_m, y_pred_m, 'test/', threshold)
        m_data.metrics.dump(args.data_result_dir, None, None, 100)
    logger.info(f'Computation done in {(time.time() - t1):8.2f} seconds')
    return 0


# noinspection DuplicatedCode
def fa_test_zshot(args) -> int:
    """
    ./mulabel fa test_zshot -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)

    train_coll_name = args.collection + '_train'
    test_coll_name = args.collection + '_test'
    data_as_dicts, _ = load_data(args, test_coll_name)

    model_data = init_model_data(args, labeler, faiss.METRIC_L2, 'zshot', models)

    t0 = time.time()
    for chunk in tqdm(chunk_data(data_as_dicts, chunk_size=384), desc='Processing zero shot eval.'):
        texts = [item['text'] for item in chunk]
        yl_true = np.array(labeler.vectorize([item['label'] for item in chunk]))
        for model_name, data in model_data.items():
            # Generate embeddings for the batch of texts
            query_vectors = data.embedder.embed(texts)  # Assuming the embedder supports batch processing
            query_vectors = query_vectors.astype(np.float32)  # Ensure correct dtype

            # Search for the topk nearest neighbors for all query vectors in the batch
            # noinspection PyArgumentList
            sim, indices = data.index.search(query_vectors, 1)  # Batched search
            yl_pred = data.train_data['y_true'][indices].squeeze()
            data.y_pred.extend(yl_pred)
            data.y_true.extend(yl_true)

    logger.info(f'Measured performance in {(time.time() - t0):8.2f} seconds')
    logger.info(f'Computing metrics')
    for m_name, m_item in model_data.items():
        y_true_m, y_pred_m = filter_metrics(args, labeler, m_item.y_true, m_item.y_pred)
        m_item.metrics(y_true_m, y_pred_m, 'test/')
        m_item.metrics.dump(args.data_result_dir, None, None, 100)

    logger.info(f'Computation done in {(time.time() - t0):8.2f} seconds')
    return 0


# noinspection DuplicatedCode
def fa_test_mlknn(args) -> int:
    """
    ./mulabel fa test_mlknn -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte
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
    for m_name, m_item in model_data.items():
        embeddings = m_item.embeddings
        y_true = m_item.train_data['y_true']
        mlknn = MLkNN(m_name, topk, 1, knn_search)
        mlknn.fit(embeddings, y_true)
        m_item.mlknn = mlknn

    train_coll_name = args.collection + '_train'
    test_coll_name = args.collection + '_test'
    data_as_dicts, _ = load_data(args, test_coll_name)

    t0 = time.time()
    for chunk in tqdm(chunk_data(data_as_dicts, chunk_size=384), desc='Processing ML-KNN eval.'):
        texts = [item['text'] for item in chunk]
        yl_true = np.array(labeler.vectorize([item['label'] for item in chunk]))
        for model_name, data in model_data.items():
            # Generate embeddings for the batch of texts
            query_vectors = data.embedder.embed(texts)  # Assuming the embedder supports batch processing
            query_vectors = query_vectors.astype(np.float32)  # Ensure correct dtype

            predictions, probabilities = data.mlknn.predict(query_vectors)
            data.y_pred.extend(probabilities)
            data.y_true.extend(yl_true)

    logger.info(f'Measured performance in {(time.time() - t0):8.2f} seconds')
    logger.info(f'Computing metrics')
    for m_name, m_item in model_data.items():
        y_true_m, y_pred_m = filter_metrics(args, labeler, m_item.y_true, m_item.y_pred)
        m_item.metrics(y_true_m, y_pred_m, 'test/', threshold)
        m_item.metrics.dump(args.data_result_dir, None, None, 100)

    logger.info(f'Computation done in {(time.time() - t0):8.2f} seconds')
    return 0
