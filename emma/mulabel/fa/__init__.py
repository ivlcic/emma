import ast
import json
import os
import logging
import random
import time

from argparse import ArgumentParser
from typing import Dict, Any, List, Callable, Union, LiteralString, Tuple

import numpy as np
import pandas as pd
import torch
import faiss

from FlagEmbedding import BGEM3FlagModel
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel


from ..gte import GTEEmbedding
from ..kalm import KaLMEmbedding
from ..mlknn import MLkNN, MLkNNAlt
from ...core.args import CommonArguments
from ...core.labels import Labeler
from ...core.metrics import Metrics
from ...core.models import retrieve_model_name_map
from ..tokenizer import get_segmenter
from ..utils import (__supported_languages, __supported_passage_sizes, __label_split_names,
                     compute_arg_collection_name, load_add_corpus_part,
                     init_labeler, filter_metrics)

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


# noinspection DuplicatedCode
def _load_data(arg, coll: str) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    data_file_name = os.path.join(arg.data_in_dir, f'{coll}.csv')
    if not os.path.exists(data_file_name):
        data_file_name = os.path.join(arg.data_out_dir, f'{coll}.csv')
    logger.info(f'Loading data from {data_file_name}...')
    data_df = load_add_corpus_part(data_file_name, 'label')
    if 'lrp' in coll and 'passage_targets' in data_df.columns:
        data_df['passage_targets'] = data_df['passage_targets'].apply(ast.literal_eval)
    return data_df.to_dict(orient='records'), data_df


def _chunk_data(data_list, chunk_size=500):
    """Generator function to yield data in chunks"""
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i + chunk_size]


def _init_segmenters(args) -> Any:
    tokenizers = {}
    for lang in args.lang:
        tokenizers[lang] = get_segmenter(lang, args.tmp_dir)
    return tokenizers


def _init_ebd_models(args) -> Dict[str, Callable[[Union[str, List[str]]], np.ndarray]]:
    logger.info(f'Loading embedding models ...')
    if 'ptm_models' not in args or not args.ptm_models:
        args.ptm_models = retrieve_model_name_map.keys()
    else:
        args.ptm_models = args.ptm_models.split(',')
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ptm_name, name in retrieve_model_name_map.items():
        if ptm_name not in args.ptm_models:
            continue
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

        if ptm_name == 'gte':
            gmodel = GTEEmbedding(name)

            # noinspection PyTypeChecker
            def gte_embed(text_to_embed: Union[str, List[str]]):
                return gmodel.encode(text_to_embed)['dense_embeddings']

            models[ptm_name] = gte_embed

        if ptm_name == 'kalm_v15':
            kmodel = KaLMEmbedding(name)

            # noinspection PyTypeChecker
            def kalm_embed(text_to_embed: Union[str, List[str]]):
                return kmodel.encode(text_to_embed)['dense_embeddings']

            models[ptm_name] = kalm_embed

    return models


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
    data_as_dicts, _ = _load_data(args, 'lrp_' + args.collection + '_train')  # we load the train data
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

    labeler = init_labeler(args)
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


def fa_lrp_extract(args) -> int:
    """
    ./mulabel fa lrp_extract -c mulabel -l sl --public
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir
    compute_arg_collection_name(args)
    labeler = init_labeler(args)
    all_labels = set(labeler.labels)
    for t in ['train', 'dev', 'test']:
        _, df = _load_data(args, f'lrp_{args.collection}_{t}')  # we load the data
        df = df[(df['passage_cat'] == 0) | (df['passage_cat'] == args.lrp_size)]
        filtered_dicts = df.to_dict(orient='records')
        filtered_data = []
        for f in filtered_dicts:  # remove samples with labels not in all_labels
            f['label'] = [label for label in f['label'] if label in all_labels]
            if len(f['label']) > 0:
                filtered_data.append(f)

        df = pd.DataFrame(filtered_data)
        df.to_csv(
            os.path.join(args.data_in_dir, f'lrp-{args.lrp_size}_{args.collection}_{t}.csv'),
            index=False, encoding='utf-8'
        )

    return 0


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
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = _init_ebd_models(args)
    labeler = init_labeler(args)
    data_as_dicts, _ = _load_data(args, args.collection + '_train')  # we load the train data

    inputs: Dict[str: Dict[str, Any]] = {}
    for model_name in models:
        inputs[model_name] = {}
        inputs[model_name]['samples'] = 0
        inputs[model_name]['embeddings'] = []
        inputs[model_name]['y_true'] = []

    for chunk in _chunk_data(data_as_dicts, chunk_size=384):
        texts: List[str] = [d['text'] for d in chunk]
        label_v = np.array([labeler.vectorize([d['label']]) for d in chunk])
        label_v = np.squeeze(label_v, axis=1)
        for model_name, model in models.items():
            ret = model(texts)
            batch_size = ret.shape[0]
            inputs[model_name]['embeddings'].append(ret)
            inputs[model_name]['y_true'].append(label_v)
            inputs[model_name]['samples'] += batch_size

    index_path = _get_index_path(args)
    for model_name in models:
        data_dict = inputs[model_name]
        data_dict['embeddings'] = np.vstack(inputs[model_name]['embeddings'])
        data_dict['y_true'] = np.vstack(inputs[model_name]['y_true'])
        # noinspection PyTypeChecker
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
    ./mulabel fa init_rae_v -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = _init_ebd_models(args)
    labeler = init_labeler(args)

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
        # noinspection PyTypeChecker
        np.savez_compressed(index_path + '.' + model_name + '_v.npz', **data_dict)

    return 0


def init_model_data(args,
                    labeler: Labeler,
                    index_metric: int,
                    prefix: str,
                    models: Dict[str, Callable[[Union[str, List[str]]], np.ndarray]]) -> Dict[str, Dict[str, Any]]:
    index_path = _get_index_path(args)
    # static model data
    model_data = {}
    for m_name, model in models.items():
        t0 = time.time()
        model_data[m_name] = {}
        x_data = np.load(index_path + '.' + m_name + '_x.npz')  # knowledge input embeddings matrix
        v_data = {'embeddings': np.array([]), 'y_id': np.array([])}
        v_data_path = index_path + '.' + m_name + '_v.npz'
        if os.path.exists(v_data_path):
            v_data = np.load(v_data_path)  # knowledge label descr embeddings matrix
            model_data[m_name]['v_data_path'] = v_data_path
            prefix = prefix + '-v_'  # mark metrics that we used label descr. embeddings and label matrix also
        else:
            model_data[m_name]['v_data_path'] = None
        model_data[m_name]['x_data'] = x_data
        model_data[m_name]['v_data'] = v_data

        suffix = ''
        if args.test_l_class != 'all':
            suffix = '_' + args.test_l_class
        m = Metrics(f'{prefix}_{m_name}_{args.collection}{suffix}', labeler.get_type_code())

        if os.path.exists(v_data_path):
            # stack training data embeddings with label description embeddings
            k_embeddings = np.vstack([x_data['embeddings'], v_data['embeddings']])  # knowledge embeddings matrix
        else:
            # no label description embeddings, use just training data embeddings
            k_embeddings = x_data['embeddings']

        k_dim = np.shape(k_embeddings)[0]
        model_data[m_name]['embedder'] = model

        # init faiss index from embeddings
        dim = np.shape(k_embeddings)[1]
        index = faiss.IndexHNSWFlat(dim, 64, index_metric)
        index.hnsw.efConstruction = 500  # Controls index construction accuracy/speed trade-off
        index.hnsw.efSearch = 300  # Controls search accuracy/speed trade-off
        # noinspection PyArgumentList
        index.add(k_embeddings)

        model_data[m_name]['embedder'] = model
        model_data[m_name]['dim'] = dim
        model_data[m_name]['k_dim'] = k_dim
        model_data[m_name]['index'] = index
        model_data[m_name]['y_true'] = []  # collect batch ground truth values
        model_data[m_name]['y_pred'] = []  # collect batch predictions
        model_data[m_name]['metrics'] = m
        logger.info(f'Loaded {m_name} data in {(time.time() - t0):8.2f} seconds')
    return model_data


def fa_hard_neg(args) -> int:
    """
    ./mulabel fa hard_neg -c mulabel -l sl
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = _init_ebd_models(args)
    labeler = init_labeler(args)

    train_coll_name = args.collection + '_train'
    test_coll_name = args.collection + '_test'
    data_as_dicts, _ = _load_data(args, train_coll_name)

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
            y_true = m_data['x_data']['y_true']
            yb_true = y_true[start_idx:end_idx]  # current batch
            query_vectors = m_data['x_data']['embeddings'][start_idx:end_idx]
            query_vectors = query_vectors.astype(np.float32)  # Ensure correct dtype

            batch_sim, batch_indices = m_data['index'].search(query_vectors, k + 1)

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
    models = _init_ebd_models(args)
    labeler = init_labeler(args)

    train_coll_name = args.collection + '_train'
    test_coll_name = args.collection + '_test'
    data_as_dicts, _ = _load_data(args, test_coll_name)

    model_data = init_model_data(args, labeler, faiss.METRIC_L2, 'raexmc', models)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # subject to grid search
    # lamb = 0.5
    lamb = 1
    topk = 50  # Number of nearest neighbors to retrieve
    threshold = 0.5  # Probability to classify as a positive
    for m_name, m_item in model_data.items():
        # knowledge values matrix
        if m_item['v_data_path'] is not None:
            lamb = 0.5
            threshold = 0.3
            k_v = np.vstack([m_item['x_data']['y_true'] * lamb, m_item['v_data']['y_id'] * (1 - lamb)])
        else:
            k_v = m_item['x_data']['y_true'] * lamb
        m_item['temperature'] = 0.04  # 1 / math.sqrt(m_item['dim'])  # 1 / sqrt(dim)
        m_item['topk'] = topk
        m_item['values'] = torch.from_numpy(k_v.astype(np.float32)).to(device)

    t1 = time.time()
    for chunk in tqdm(_chunk_data(data_as_dicts, chunk_size=384), desc='Processing RAE-XML complete eval.'):
        texts = [item['text'] for item in chunk]
        yl_true = np.array(labeler.vectorize([item['label'] for item in chunk]))
        for model_name, data in model_data.items():
            # Generate embeddings for the batch of texts
            query_vectors = data['embedder'](texts)  # Assuming the embedder supports batch processing
            query_vectors = query_vectors.astype(np.float32)  # Ensure correct dtype

            # Search for the topk nearest neighbors for all query vectors in the batch
            sim, indices = data['index'].search(query_vectors, data['topk'])  # Batched search
            sim = torch.from_numpy(sim).to(device)
            # Convert similarities to probability distribution using softmax
            sim = F.softmax(-sim / data['temperature'], dim=-1)
            # Initialize qKT tensor for the batch
            batch_size = len(texts)
            qKT = torch.zeros((batch_size, data['k_dim']), dtype=torch.float32).to(device)
            # Assign values to specific indices for each item in the batch
            for i in range(batch_size):
                qKT[i, indices[i]] = sim[i]

            # Compute probabilities using matrix multiplication
            probabilities = torch.matmul(qKT, data['values'])
            yl_prob = (probabilities > threshold).cpu().numpy()
            data['y_pred'].extend(yl_prob)
            data['y_true'].extend(yl_true)

    logger.info(f'Measured performance in {(time.time() - t1):8.2f} seconds')
    logger.info(f'Computing metrics')
    for m_name, m_item in model_data.items():
        y_true_m, y_pred_m = filter_metrics(args, labeler, m_item['y_true'], m_item['y_pred'])
        m_item['metrics'](y_true_m, y_pred_m, 'test/', threshold)
        m_item['metrics'].dump(args.data_result_dir, None, None, 100)
    logger.info(f'Computation done in {(time.time() - t1):8.2f} seconds')
    return 0


# noinspection DuplicatedCode
def fa_test_zshot(args) -> int:
    """
    ./mulabel fa test_zshot -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = _init_ebd_models(args)
    labeler = init_labeler(args)

    train_coll_name = args.collection + '_train'
    test_coll_name = args.collection + '_test'
    data_as_dicts, _ = _load_data(args, test_coll_name)

    model_data = init_model_data(args, labeler, faiss.METRIC_L2, 'zshot', models)

    t0 = time.time()
    for chunk in tqdm(_chunk_data(data_as_dicts, chunk_size=384), desc='Processing RAE-XML complete eval.'):
        texts = [item['text'] for item in chunk]
        yl_true = np.array(labeler.vectorize([item['label'] for item in chunk]))
        for model_name, data in model_data.items():
            # Generate embeddings for the batch of texts
            query_vectors = data['embedder'](texts)  # Assuming the embedder supports batch processing
            query_vectors = query_vectors.astype(np.float32)  # Ensure correct dtype

            # Search for the topk nearest neighbors for all query vectors in the batch
            sim, indices = data['index'].search(query_vectors, 1)  # Batched search
            yl_pred = data['x_data']['y_true'][indices].squeeze()
            data['y_pred'].extend(yl_pred)
            data['y_true'].extend(yl_true)

    logger.info(f'Measured performance in {(time.time() - t0):8.2f} seconds')
    logger.info(f'Computing metrics')
    for m_name, m_item in model_data.items():
        y_true_m, y_pred_m = filter_metrics(args, labeler, m_item['y_true'], m_item['y_pred'])
        m_item['metrics'](y_true_m, y_pred_m, 'test/')
        m_item['metrics'].dump(args.data_result_dir, None, None, 100)

    logger.info(f'Computation done in {(time.time() - t0):8.2f} seconds')
    return 0


# noinspection DuplicatedCode
def fa_test_mlknn(args) -> int:
    """
    ./mulabel fa test_mlknn -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = _init_ebd_models(args)
    labeler = init_labeler(args)

    model_data = init_model_data(args, labeler, faiss.METRIC_L2, 'mlknn', models)

    def knn_search(mn: str, queries: np.ndarray, k: int):
        # Ensure the queries are 2D
        queries = queries.reshape(-1, queries.shape[1]).astype(np.float32)

        # Perform a batched search for all query vectors
        # noinspection PyArgumentList
        index = model_data[mn]['index']
        sim, indices = index.search(queries, k=(k + 1))
        indices = np.delete(indices, 0, axis=1)  # remove first - self match

        return indices

    threshold = 0.5
    topk = 50  # Number of nearest neighbors to retrieve
    for m_name, m_item in model_data.items():
        embeddings = m_item['x_data']['embeddings']
        y_true = m_item['x_data']['y_true']
        mlknn = MLkNN(m_name, topk, 1, knn_search)
        mlknn.fit(embeddings, y_true)
        m_item['mlknn'] = mlknn

    train_coll_name = args.collection + '_train'
    test_coll_name = args.collection + '_test'
    data_as_dicts, _ = _load_data(args, test_coll_name)

    t0 = time.time()
    for chunk in tqdm(_chunk_data(data_as_dicts, chunk_size=384), desc='Processing RAE-XML complete eval.'):
        texts = [item['text'] for item in chunk]
        yl_true = np.array(labeler.vectorize([item['label'] for item in chunk]))
        for model_name, data in model_data.items():
            # Generate embeddings for the batch of texts
            query_vectors = data['embedder'](texts)  # Assuming the embedder supports batch processing
            query_vectors = query_vectors.astype(np.float32)  # Ensure correct dtype

            predictions, probabilities = data['mlknn'].predict(query_vectors)
            data['y_pred'].extend(probabilities)
            data['y_true'].extend(yl_true)

    logger.info(f'Measured performance in {(time.time() - t0):8.2f} seconds')
    logger.info(f'Computing metrics')
    for m_name, m_item in model_data.items():
        y_true_m, y_pred_m = filter_metrics(args, labeler, m_item['y_true'], m_item['y_pred'])
        m_item['metrics'](y_true_m, y_pred_m, 'test/', threshold)
        m_item['metrics'].dump(args.data_result_dir, None, None, 100)

    logger.info(f'Computation done in {(time.time() - t0):8.2f} seconds')
    return 0
