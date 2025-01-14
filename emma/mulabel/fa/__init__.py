import ast
import math
import os
import logging
import random
import time

from argparse import ArgumentParser
from typing import Dict, Any, List, Callable, Union, LiteralString

import numpy as np
import pandas as pd
import torch
import faiss

from FlagEmbedding import BGEM3FlagModel
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel


from ...core.args import CommonArguments
from ...core.labels import MultilabelLabeler, Labeler
from ...core.metrics import Metrics
from ...core.models import retrieve_model_name_map
from ...core.wandb import initialize_run
from ..tokenizer import get_segmenter
from ..utils import (__supported_languages, __supported_passage_sizes, __label_split_names, __label_splits,
                     compute_arg_collection_name, load_add_corpus_part, parse_arg_passage_sizes,
                     split_csv_by_frequency)

logger = logging.getLogger('mulabel.fa')


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
        '--calib_max', type=int, help=f'Max number of labels to calibrate on.', default=-1
    )
    parser.add_argument(
        '--test_l_class', type=str, help=f'Test specified label class.',
        choices=['all'].extend(__label_split_names), default='all'
    )


# noinspection DuplicatedCode
def _load_data(arg, coll: str) -> List[Dict[str, Any]]:
    data_file_name = os.path.join(arg.data_in_dir, f'{coll}.csv')
    if not os.path.exists(data_file_name):
        data_file_name = os.path.join(arg.data_out_dir, f'{coll}.csv')
    logger.info(f'Loading data from {data_file_name}...')
    data_df = load_add_corpus_part(data_file_name, 'label')
    if 'lrp' in coll and 'passage_targets' in data_df.columns:
        data_df['passage_targets'] = data_df['passage_targets'].apply(ast.literal_eval)
    return data_df.to_dict(orient='records')


def _chunk_data(data_list, chunk_size=500):
    """Generator function to yield data in chunks"""
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i + chunk_size]


def _init_segmenters(args) -> Any:
    tokenizers = {}
    for lang in args.lang:
        tokenizers[lang] = get_segmenter(lang, args.tmp_dir)
    return tokenizers


# noinspection DuplicatedCode
def _init_ebd_models() -> Dict[str, Callable[[Union[str, List[str]]], np.ndarray]]:
    logger.info(f'Loading embedding models ...')
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ptm_name, name in retrieve_model_name_map.items():
        if ptm_name == 'bge_m3':

            bmodel = BGEM3FlagModel(
                name, use_fp16=True, device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            def bge_m3_embed(text_to_embed: Union[str, List[str]]):
                return bmodel.encode(text_to_embed)['dense_vecs']

            models[ptm_name] = bge_m3_embed

        if ptm_name == 'jina3':
            jmodel = AutoModel.from_pretrained(
                name, trust_remote_code=True
            )
            jmodel.to(device)

            def jina3_embed(text_to_embed: Union[str, List[str]]):
                return jmodel.encode(text_to_embed, task='text-matching', show_progress_bar=False)

            models[ptm_name] = jina3_embed

    return models


# noinspection DuplicatedCode
def _init_labeler(args) -> Labeler:
    labels_file_name = os.path.join(args.data_in_dir, f'{args.collection}_labels.csv')
    if not os.path.exists(labels_file_name) and 'lrp' in args.collection:
        tmp = args.collection.replace('lrp_', '')
        labels_file_name = os.path.join(args.data_in_dir, f'{tmp}_labels.csv')
        if not os.path.exists(labels_file_name) and 'lrp' in args.collection:
            raise ValueError(f'Missing labels file [{labels_file_name}]')

    with open(labels_file_name, 'r') as l_file:
        all_labels = [line.split(',')[0].strip() for line in l_file]
    if all_labels[0] == 'label':
        all_labels.pop(0)
    labeler = MultilabelLabeler(all_labels)
    labeler.fit()
    return labeler


def _load_labels(split_dir, corpus: str, splits: List[int], names: List[str]) -> Dict[str, Dict[str, int]]:
    l_file_path = os.path.join(split_dir, f'{corpus}_labels.csv')
    if os.path.exists(l_file_path):
        return split_csv_by_frequency(l_file_path, splits, names)


def __find_label_in_text(kwes: List[str], doc) -> List[str]:
    passages = []
    for sentence in doc.sentences:
        for kwe in kwes:
            if kwe in sentence.text:
                passages.append(sentence.text)
    return passages


def fa_init_pseudo_labels(args) -> int:
    """
    ./mulabel fa init_pseudo_labels -c mulabel -l sl --public
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    segmenters = _init_segmenters(args)

    # now we construct pseudo label descriptions
    label_dict: Dict[str, Dict['str', Any]] = {}
    data_as_dicts = _load_data(args, 'lrp_' + args.collection + '_train')  # we load the train data
    for item in data_as_dicts:
        if item['passage_cat'] != 0 and item['passage_cat'] != 1:
            continue
        for k, label in enumerate(item['label']):
            if label not in label_dict:
                label_dict[label] = {
                    'label': label,
                    'label_info': item['label_info'][k],
                    'texts': [], 'passages': []
                }
            if item['passage_cat'] == 1:
                label_dict[label]['passages'].append(item['text'])
            else:
                passages = []
                if item['lang'] in segmenters:
                    search = []
                    if 'kwe' in item['label_info'][k]:
                        search.extend([k['value'] for k in item['label_info'][k]['kwe'] if 'value' in k])
                    if 'name' in item['label_info'][k]:
                        search.append(item['label_info'][k]['name'])

                    if search:
                        if 'doc' not in item:
                            segmenter = segmenters[item['lang']]
                            item['doc'] = segmenter(item['text'])
                        passages = __find_label_in_text(search, item['doc'])
                if passages:
                    label_dict[label]['passages'].extend(passages)
                else:
                    label_dict[label]['texts'].append(item['text'])

    labeler = _init_labeler(args)
    labels_df_data = []
    for id, label in labeler.ids_to_labels().items():
        if label not in label_dict:
            logger.warning(f'Missing any text for label {label}')
            continue
        label_data = label_dict[label]
        label_data['id'] = id
        labels_df_data.append(label_data)

    labels_df = pd.DataFrame(labels_df_data)
    labels_df.to_csv(os.path.join(
        args.data_in_dir, f'{args.collection}_labels_descr.csv'), index=False, encoding='utf-8')

    return 0


def _get_index_path(args) -> LiteralString | str | bytes:
    index_path = os.path.join(args.data_result_dir, 'index')
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    index_path = os.path.join(args.data_result_dir, 'index', args.collection + '_train')
    return index_path


def fa_init_rae_x(args) -> int:
    """
    ./mulabel fa init_knowledge_x -c mulabel -l sl --public
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = _init_ebd_models()
    labeler = _init_labeler(args)
    data_as_dicts = _load_data(args, args.collection + '_train')  # we load the train data

    inputs: Dict[str: Dict[str, Any]] = {}
    for model_name in models:
        inputs[model_name] = {}
        inputs[model_name]['samples'] = 0
        inputs[model_name]['embeddings'] = []
        inputs[model_name]['y_true'] = []

    for chunk in _chunk_data(data_as_dicts, chunk_size=384):
        texts: List[str] = [d['text'] for d in chunk]
        labels = np.array([labeler.vectorize([d['label']]) for d in chunk])
        labels = np.squeeze(labels, axis=1)
        for model_name, model in models.items():
            ret = model(texts)
            batch_size = ret.shape[0]
            inputs[model_name]['embeddings'].append(ret)
            inputs[model_name]['y_true'].append(labels)
            inputs[model_name]['samples'] += batch_size

    index_path = _get_index_path(args)
    for model_name in models:
        data_dict = inputs[model_name]
        data_dict['embeddings'] = np.vstack(inputs[model_name]['embeddings'])
        data_dict['y_true'] = np.vstack(inputs[model_name]['y_true'])
        np.savez_compressed(index_path + '.' + model_name + '_x.npz', **data_dict)

    return 0


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


def fa_init_rae_v(args) -> int:
    """
    ./mulabel fa init_knowledge_v -c mulabel -l sl --public
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = _init_ebd_models()
    labeler = _init_labeler(args)

    # read label descriptions / passages
    label_descr_file_path = os.path.join(args.data_in_dir, f'{args.collection}_labels_descr.csv')
    if not os.path.exists(label_descr_file_path):
        logger.warning(f'Missing label description file [{label_descr_file_path}]. '
                       f'Run [./mulabel fa init_pseudo_labels -c mulabel -l sl --public] or similar first!')
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
        labels[model_name]['embeddings'] = []
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

    for chunk in _chunk_data(texts, chunk_size=384):
        for model_name, model in models.items():
            ret = model(chunk)
            labels[model_name]['embeddings'].append(ret)

    index_path = _get_index_path(args)
    for model_name in models:
        data_dict = labels[model_name]
        data_dict['embeddings'] = np.vstack(labels[model_name]['embeddings'])
        data_dict['y_id'] = np.identity(num_labels)
        np.savez_compressed(index_path + '.' + model_name + '_v.npz', **data_dict)

    return 0


def _filter_data_labels(labeler: Labeler, data_items, target_indices, filter_labels):
    texts = [item['text'] for item in data_items]
    labels = [item['label'] for item in data_items]
    yl_true = np.array(labeler.vectorize(labels))
    mask = np.zeros(labeler.num_labels, dtype=bool)
    if filter_labels:  # keep only desired labels and zero-out undesired labels
        mask[target_indices] = True
        yl_true = yl_true * mask
        mask_non_zero = ~np.all(yl_true == 0, axis=1)
        texts = [value for idx, value in enumerate(texts) if mask_non_zero[idx]]
        yl_true = yl_true[mask_non_zero]

    return texts, yl_true, mask


# noinspection DuplicatedCode
def fa_test_rae(args) -> int:
    """
    ./mulabel fa test -c mulabel -l sl --public
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = _init_ebd_models()
    labeler = _init_labeler(args)

    suffix = ''
    if args.test_l_class != 'all':
        suffix = '_' + args.test_l_class

    metrics = {}
    y_true = {}
    y_pred = {}
    for name in models:
        metrics[name] = Metrics(f'raexmc_{name}_{args.collection}{suffix}', labeler.get_type_code())
        y_true[name] = []
        y_pred[name] = []

    train_coll_name = args.collection + '_train'
    test_coll_name = args.collection + '_test'
    data_as_dicts = _load_data(args, test_coll_name)

    target_indices = []
    filter_labels = False
    if args.test_l_class != 'all':
        label_classes = _load_labels(args.data_in_dir, args.collection, __label_splits, __label_split_names)
        target_labels = label_classes[args.test_l_class]
        target_indices = [labeler.encoder.classes_.tolist().index(label) for label in target_labels.keys()]
        filter_labels = True

    index_path = _get_index_path(args)
    # static model data
    model_data = {}
    for model_name, model in models.items():
        t0 = time.time()
        model_data[model_name] = {}
        x_data = np.load(index_path + '.' + model_name + '_x.npz')  # knowledge input embeddings matrix
        v_data = {'embeddings': np.array([]), 'y_id': np.array([])}
        v_data_path = index_path + '.' + model_name + '_v.npz'
        if os.path.exists(v_data_path):
            v_data = np.load(v_data_path)  # knowledge label descr embeddings matrix
        model_data[model_name]['x_data'] = x_data
        model_data[model_name]['v_data'] = v_data

        k_embeddings = np.vstack([x_data['embeddings'], v_data['embeddings']])  # knowledge embeddings matrix
        k_dim = np.shape(k_embeddings)[0]
        model_data[model_name]['embedder'] = model

        dim = np.shape(k_embeddings)[1]
        index = faiss.IndexHNSWFlat(dim, 64)
        index.hnsw.efConstruction = 500  # Controls index construction accuracy/speed trade-off
        index.hnsw.efSearch = 300  # Controls search accuracy/speed trade-off
        index.add(k_embeddings)
        model_data[model_name]['embedder'] = model
        model_data[model_name]['dim'] = dim
        model_data[model_name]['k_dim'] = k_dim
        model_data[model_name]['index'] = index
        logger.info(f'Loaded {model_name} data in {(time.time() - t0):8.2f} seconds')

    # subject to grid search
    lamb = 0.5
    topk = 100  # Number of nearest neighbors to retrieve
    for m_name, m_item in model_data.items():
        # knowledge values matrix
        k_v = np.vstack([m_item['x_data']['y_true'] * lamb, m_item['v_data']['y_id'] * (1 - lamb)])
        m_item['temperature'] = 1 / math.sqrt(m_item['dim'])  # 1 / sqrt(dim)
        m_item['topk'] = topk
        m_item['values'] = torch.from_numpy(k_v.astype(np.float32))

    t1 = time.time()
    for chunk in tqdm(_chunk_data(data_as_dicts, chunk_size=64), desc='Processing RAE-XML complete eval.'):
        texts, yl_true, mask = _filter_data_labels(labeler, chunk, target_indices, filter_labels)
        if len(texts) == 0:
            continue

        for model_name, data in model_data.items():
            # Generate embeddings for the batch of texts
            query_vectors = data['embedder'](texts)  # Assuming the embedder supports batch processing
            query_vectors = query_vectors.astype(np.float32)  # Ensure correct dtype

            # Search for the topk nearest neighbors for all query vectors in the batch
            sim, indices = data['index'].search(query_vectors, data['topk'])  # Batched search
            # Convert similarities to probabilities using softmax
            sim = F.softmax(torch.from_numpy(-sim / data['temperature']), dim=-1)
            # Initialize qKT tensor for the batch
            batch_size = len(texts)
            qKT = torch.zeros((batch_size, data['k_dim']), dtype=torch.float32)
            # Assign values to specific indices for each item in the batch
            for i in range(batch_size):
                qKT[i, indices[i]] = sim[i]

            # Compute probabilities using matrix multiplication
            probabilities = torch.matmul(qKT, data['values'])
            yl_pred = (probabilities > 0.2).int().numpy()
            # Un-vectorize predictions and collect results
            if filter_labels:
                yl_pred = yl_pred * mask
            y_pred[model_name].extend(yl_pred)
            y_true[model_name].extend(yl_true)

    logger.info(f'Measured performance in {(time.time() - t1):8.2f} seconds')
    logger.info(f'Computing metrics')
    for model_name in models:
        y_true_m = np.array(y_true[model_name])
        y_pred_m = np.array(y_pred[model_name])
        metrics[model_name](y_true_m, y_pred_m, 'test/')
        metrics[model_name].dump(args.data_result_dir, None, None)
    logger.info(f'Computation done in {(time.time() - t0):8.2f} seconds')
    return 0
