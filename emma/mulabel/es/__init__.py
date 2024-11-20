import ast
import os
import logging
from datetime import datetime

from argparse import ArgumentParser
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics import roc_curve
from tqdm import tqdm

from core.labels import Labeler
from ...core.args import CommonArguments
from ...core.labels import MultilabelLabeler
from ...core.metrics import Metrics
from ...core.wandb import initialize_run
from ..tokenizer import get_segmenter
from ..utils import __supported_languages, compute_arg_collection_name, load_add_corpus_part
from .utils import load_data, find_similar

logger = logging.getLogger('mulabel.es')

es_logger = logging.getLogger("elastic_transport.transport")
es_logger.setLevel(logging.WARN)

CLIENT_URL = "http://localhost:9266/"


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
        '--passage_size', help='When calibrating use passage_size',
        type=int, default=1, choices=[1, 3, 5, 7, 9, 0]
    )
    parser.add_argument(
        '--run_id', type=int, help=f'Run id for marking consecutive runs.', default=0
    )


def es_init(arg) -> int:
    """
    ./mulabel db init -c lrp_mulabel
    ./mulabel db init -c lrp_mulabel -l sl
    ./mulabel db init -c lrp_mulabel -l sl,sr
    """
    compute_arg_collection_name(arg)
    client = Elasticsearch(CLIENT_URL)
    try:
        # create collection if it doesn't exist
        if not client.indices.exists(index=arg.collection):
            logger.info('Collection [%s] does not exist. Creating...', arg.collection)
            client.indices.create(index=arg.collection)
            logger.info('Collection [%s] created successfully.', arg.collection)
        else:
            logger.warning('Collection [%s] already exists.', arg.collection)
    finally:
        client.close()
    return 0


def es_drop(arg) -> int:
    """
    ./mulabel es drop -c lrp_mulabel
    ./mulabel es drop -c lrp_mulabel -l sl,sr
    """
    compute_arg_collection_name(arg)
    client = Elasticsearch(CLIENT_URL)
    try:
        if client.indices.exists(index=arg.collection):
            client.indices.delete(index=arg.collection)
            logger.info('Collection [%s] deleted.', arg.collection)
        else:
            logger.warning('Collection [%s] does not exist.', arg.collection)
    finally:
        client.close()
    return 0


def _load_data(arg, coll: str) -> List[Dict[str, Any]]:
    data_file_name = os.path.join(arg.data_in_dir, f'{coll}.csv')
    if not os.path.exists(data_file_name):
        data_file_name = os.path.join(arg.data_out_dir, f'{coll}.csv')
    data_df = load_add_corpus_part(data_file_name, 'label')
    if 'lrp' in coll and 'passage_targets' in data_df.columns:
        data_df['passage_targets'] = data_df['passage_targets'].apply(ast.literal_eval)
    return data_df.to_dict(orient='records')


def _chunk_data(data_list, chunk_size=500):
    """Generator function to yield data in chunks"""
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i + chunk_size]


def _prepare_documents(coll: str, models, data_chunk):
    """Prepare documents for bulk indexing"""
    for data_item in data_chunk:
        # logger.info('Processing item [%s]', data_item)
        for model_name, model in models.items():
            data_item['m_' + model_name] = model(data_item['text'])[0].tolist()
        op = {
            '_op_type': 'index',
            '_index': f'{coll}',
            '_id': data_item['uuid'] if 'uuid' in data_item else data_item['a_uuid'],
            '_source': data_item
        }
        yield op


# noinspection DuplicatedCode
def es_pump(args) -> int:
    """
    Pumps the data in elastic search index
    ./mulabel es pump -c lrp_mulabel
    ./mulabel es pump -c lrp_mulabel -l sl,sr

    ./mulabel es drop -c lrp_mulabel -l sl --public
    ./mulabel es init -c lrp_mulabel -l sl --public
    ./mulabel es pump -c lrp_mulabel -l sl --public

    ./mulabel es drop -c mulabel -l sl --public
    ./mulabel es init -c mulabel -l sl --public
    ./mulabel es pump -c mulabel -l sl --public

    ./mulabel es pump -c mulabel -l sl --public --seed_only
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    tokenizers = {}
    for lang in args.lang:
        tokenizers[lang] = get_segmenter(lang, args.tmp_dir)
    bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def bge_m3_embed(text_to_embed: str):
        return bge_m3_model.encode([text_to_embed])['dense_vecs']

    models = {
        'bge_m3': bge_m3_embed
    }

    data_as_dicts = _load_data(args, args.collection)

    client = Elasticsearch(CLIENT_URL)
    try:
        if not client.indices.exists(index=args.collection):
            logger.warning('Collection [%s] does not exist.', args.collection)
            return 1

        total_indexed = 0
        total_failed = 0
        for chunk in _chunk_data(data_as_dicts, chunk_size=1000):
            success, failed = bulk(
                client=client,
                actions=_prepare_documents(args.collection, models, chunk),
                request_timeout=60
            )

            total_indexed += success
            total_failed += len(failed) if failed else 0
            logger.info(f"Processed chunk: {success} succeeded, {len(failed) if failed else 0} failed")
    finally:
        client.close()

    return 0


# noinspection DuplicatedCode
def es_dedup(args) -> int:
    """
    ./mulabel es pump -c lrp_mulabel
    ./mulabel es pump -c lrp_mulabel -l sl,sr

    ./mulabel es drop -c lrp_mulabel -l sl --public
    ./mulabel es init -c lrp_mulabel -l sl --public
    ./mulabel es pump -c lrp_mulabel -l sl --public

    ./mulabel es drop -c mulabel -l sl --public
    ./mulabel es init -c mulabel -l sl --public
    ./mulabel es pump -c mulabel -l sl --public
    ./mulabel es dedup -c mulabel -l sl --public
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir
    compute_arg_collection_name(args)
    start_date = datetime(2022, 12, 31)
    end_date = datetime(2023, 6, 2)

    state = {'count': 0, 'duplicates': set(), 'dup_map': {}, 'doc': {}}
    duplicates = []
    client = Elasticsearch(CLIENT_URL)
    try:
        def on_similar(data_item: Dict[str, Any], score: float) -> bool:
            if score > 0.99:
                q_a_uuid = state['doc']['a_uuid']
                v_a_uuid = data_item['a_uuid']
                # logger.info(
                #    'Article [%s] [%s][%s] is very similar to [%s][%s]',
                #    state['count'], q_a_uuid, state['doc']['date'], v_a_uuid, data_item['date']
                # )
                # logger.info('Similar texts: \n\n[%s]\n\n[%s]', state['doc']['text'], data_item['text'])
                if q_a_uuid not in state['dup_map']:
                    state['dup_map'][q_a_uuid] = [v_a_uuid]
                else:
                    state['dup_map'][q_a_uuid].append(v_a_uuid)
                state['duplicates'].add(v_a_uuid)
                del data_item['m_bge_m3']
                data_item['similar_uuid'] = q_a_uuid
                data_item['similar_id'] = state['doc']['a_id']
                # data_item['similar_text'] = state['doc']['text']
                duplicates.append(data_item)
            return True

        def on_item(data_item: Dict[str, Any]) -> bool:
            state['count'] += 1
            state['doc'] = data_item
            if data_item['a_uuid'] in state['duplicates']:
                return True
            find_similar(client, args.collection, data_item['a_uuid'], data_item['m_bge_m3'], on_similar)
            return True

        load_data(client, args.collection, start_date, end_date, on_item)
    finally:
        client.close()

    lrp = 'lrp' in args.collection
    data_file_name = os.path.join(
        args.data_out_dir, f'lrp_{args.collection}.csv' if lrp else f'{args.collection}_duplicates.csv'
    )
    df = pd.DataFrame(duplicates)
    df.to_csv(data_file_name, index=False, encoding='utf8')
    return 0


def _init_task(args, name, ptm_name, ptm_alias) -> Any:
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir
    args.ptm_alias = 'bge_m3'
    args.ptm_name = 'BAAI/bge-m3'

    tags = [
        args.collection_conf, args.ptm_name, args.collection
    ]
    if 'lang_conf' in args and args.lang_conf:
        tags.extend(args.lang_conf.split(','))
    if args.public:
        tags.append('public')
    if args.seed_only:
        tags.append('seed_labels')

    params = {
        'job_type': 'retrieval_test',
        'name': name,
        'run_id': name + '_' + args.ptm_alias + '@' + str(args.run_id),
        'run_group': args.collection_conf,
        'tags': tags,
        'conf': {
            'ptm_alias': args.ptm_name,
            'lang': args.lang_conf,
            'corpus': args.collection
        }
    }
    return initialize_run(**params)


def _init_segmenters(args) -> Any:
    tokenizers = {}
    for lang in args.lang:
        tokenizers[lang] = get_segmenter(lang, args.tmp_dir)
    return tokenizers


def _init_ebd_models(args) -> Any:
    bge_m3_model = BGEM3FlagModel(args.ptm_name, use_fp16=True)

    def bge_m3_embed(text_to_embed: str):
        return bge_m3_model.encode([text_to_embed])['dense_vecs']

    models = {
        args.ptm_alias: bge_m3_embed
    }
    return models


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


# noinspection DuplicatedCode
def es_test_bge_m3(args) -> int:
    """
    ./mulabel es test_bge_m3 -c mulabel -l sl
    ./mulabel es test_bge_m3 -c mulabel -l sl --public
    ./mulabel es test_bge_m3 -c mulabel -l sl --public --seed_only
    """
    compute_arg_collection_name(args)
    output_name = 'zshot_' + args.collection
    run = _init_task(args, output_name, 'BAAI/bge-m3', 'bge_m3')
    labeler = _init_labeler(args)
    metrics = Metrics('zshot_' + args.collection, labeler.get_type_code())
    models = _init_ebd_models(args)
    model = models[args.ptm_alias]

    train_coll_name = args.collection + '_train'
    test_coll_name = args.collection + '_test'
    data_as_dicts = _load_data(args, test_coll_name)

    state = {'doc': {}, 'count': 0}
    y_true = []
    y_pred = []
    client = Elasticsearch(CLIENT_URL)
    try:
        if not client.indices.exists(index=train_coll_name):
            logger.warning('Collection [%s] does not exist.', train_coll_name)
            return 1

        def on_similar(similar_item: Dict[str, Any], score: float) -> bool:
            y_pred.append(labeler.vectorize([similar_item['label']])[0])
            y_true.append(labeler.vectorize([state['doc']["label"]])[0])
            return False

        for data_item in tqdm(data_as_dicts, desc='Processing zero shot complete eval.'):
            data_item['m_' + args.ptm_alias] = model(data_item['text'])[0].tolist()
            state['doc'] = data_item
            state['count'] += 1
            num_ret = find_similar(
                client, train_coll_name, data_item['a_uuid'], data_item[f'm_{args.ptm_alias}'], on_similar
            )
            # if state['count'] > 100:
            #    break
            if num_ret == 0:
                y_pred.append(labeler.vectorize([[]])[0])

    finally:
        client.close()

    m = metrics(np.array(y_true, dtype=float), np.array(y_pred, dtype=float), 'test/')
    metrics.dump(args.data_result_dir, None, run)
    if run is not None:
        run.log(m)

    return 0


def es_calibrate_lrp_bge_m3(args) -> int:
    compute_arg_collection_name(args)
    output_name = f'{args.collection}_calib_'
    run = _init_task(args, output_name, 'BAAI/bge-m3', 'bge_m3')
    labeler = _init_labeler(args)
    models = _init_ebd_models(args)
    model = models[args.ptm_alias]

    train_coll_name = args.collection + '_train'
    dev_coll_name = args.collection + '_dev'
    data_as_dicts = _load_data(args, train_coll_name)
    data_as_dicts.extend(_load_data(args, dev_coll_name))
    passage_sizes = [1, 3, 5, 7, 9]

    def find_optimal_threshold(y_true, y_prob):
        if len(y_true) == 0:
            return 0.0
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_j = tpr - fpr  # Youden's J statistic
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold



    client = Elasticsearch(CLIENT_URL)
    try:
        if not client.indices.exists(index=train_coll_name):
            logger.warning('Collection [%s] does not exist.', train_coll_name)
            return 1
        df = pd.DataFrame(data=data_as_dicts)
        for passage_size in passage_sizes:
            label_thresholds = {}
            for i, label in tqdm(
                    enumerate(labeler.labels), f'Processing labels for passage category {passage_size}:'
            ):
                filtered_df = df[(df['passage_cat'] == passage_size) & (df['label'].apply(lambda x: label in x))]
                state = {'doc': {}, 'count': 0, 'similar': []}
                y_true = []
                y_prob = []
                passage_data_as_dicts = filtered_df.to_dict(orient='records')
                for data_i, data_item in enumerate(passage_data_as_dicts):

                    def on_similar(similar_item: Dict[str, Any], score: float) -> bool:
                        if label not in similar_item['label']:
                            y_prob.append(score)
                            y_true.append(0)
                            return True
                        y_prob.append(score)
                        y_true.append(1)
                        state['similar'].append(similar_item)
                        return True

                    state['doc'] = data_item
                    data_item['m_' + args.ptm_alias] = model(data_item['text'])[0].tolist()
                    num_ret = find_similar(
                        client, train_coll_name, data_item['a_uuid'], data_item[f'm_{args.ptm_alias}'],
                        on_similar, size=10, passage_cat=[data_item['passage_cat']]
                    )
                    if num_ret == 0:
                        y_prob.append(0.0)
                        y_true.append(1)

                if label not in label_thresholds:
                    label_thresholds[label] = {}
                if len(y_prob) == 0:
                    label_thresholds[label]['label'] = label
                    label_thresholds[label][f'opt'] = 0.0
                    label_thresholds[label][f'min'] = 0.0
                    label_thresholds[label][f'max'] = 0.0
                    label_thresholds[label][f'mean'] = 0.0
                    label_thresholds[label][f'pos'] = 0.0
                    label_thresholds[label][f'neg'] = 0.0
                    label_thresholds[label][f'num'] = 0.0
                else:
                    y_prob = np.array(y_prob)
                    y_true = np.array(y_true)
                    tmp = y_prob * y_true
                    tmp = tmp[tmp > 0]
                    label_thresholds[label]['label'] = label
                    label_thresholds[label][f'opt'] = find_optimal_threshold(y_true, y_prob)
                    label_thresholds[label][f'min'] = np.min(tmp) if len(tmp) > 0 else 0.0
                    label_thresholds[label][f'max'] = np.max(tmp) if len(tmp) > 0 else 0.0
                    label_thresholds[label][f'mean'] = np.mean(tmp) if len(tmp) > 0 else 0.0
                    label_thresholds[label][f'pos'] = np.sum(y_true)
                    label_thresholds[label][f'neg'] = y_true.shape[0] - np.sum(y_true)
                    label_thresholds[label][f'num'] = y_true.shape[0]

                if i % 100 == 0:
                    logger.info(f'At label {i}/{label}')
            calib_cat_file_name = os.path.join(args.data_in_dir, f'{output_name}_{passage_size}.csv')
            pd.DataFrame(data=label_thresholds).to_csv(calib_cat_file_name, index=False, encoding='utf-8')
    finally:
        client.close()

    return 0


def es_test_lrp_bge_m3(args) -> int:
    """
    In development - specifically for LRP
    ./mulabel es test_lrp_bge_m3 -c lrp_mulabel -l sl
    ./mulabel es test_lrp_bge_m3 -c lrp_mulabel -l sl --public
    ./mulabel es test_lrp_bge_m3 -c lrp_mulabel -l sl --public --seed_only
    """
    compute_arg_collection_name(args)
    output_name = 'zshot_' + args.collection
    run = _init_task(args, output_name, 'BAAI/bge-m3', 'bge_m3')
    labeler = _init_labeler(args)
    metrics = Metrics('zshot_' + args.collection, labeler.get_type_code())

    segmenters = _init_segmenters(args)
    models = _init_ebd_models(args)
    model = models[args.ptm_alias]

    train_coll_name = args.collection + '_train'
    test_coll_name = args.collection + '_test'
    data_as_dicts = _load_data(args, test_coll_name)

    state = {'count': 0, 'similar': []}
    y_true = []
    y_pred = []
    client = Elasticsearch(CLIENT_URL)
    try:
        if not client.indices.exists(index=train_coll_name):
            logger.warning('Collection [%s] does not exist.', train_coll_name)
            return 1

        def on_similar(similar_item: Dict[str, Any], score: float) -> bool:
            if score > 0.80:
                for model_name, model in models.items():
                    del similar_item['m_' + model_name]
                similar_item['score'] = score
                state['similar'].append(similar_item)
            return True

        for data_item in tqdm(data_as_dicts, desc='Processing zero shot complete eval.'):
            if data_item['passage_cat'] != 1:
                continue
            state['count'] += 1
            state['similar'] = []
            if data_item['lang'] not in segmenters:
                logger.warning('Language [%s] does not exist.', data_item['lang'])
                continue

            for sentence in segmenters[data_item['lang']](data_item['text']).sentences:
                text = sentence.text
                data_item['m_' + args.ptm_alias] = model(text)[0].tolist()
                num_ret = find_similar(
                    client, train_coll_name, data_item['a_uuid'], data_item[f'm_{args.ptm_alias}'],
                    on_similar, size=50, passage_cat=[data_item['passage_cat']]
                )
                if num_ret <= 0:
                    logger.warning("kr neki")

            # if state['count'] > 100:
            #    break

    finally:
        client.close()

    m = metrics(np.array(y_true, dtype=float), np.array(y_pred, dtype=float), 'test/')
    metrics.dump(args.data_result_dir, None, run)
    if run is not None:
        run.log(m)

    return 0
