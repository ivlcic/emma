import os
import logging
from datetime import datetime

from argparse import ArgumentParser
from typing import Dict, Any, List

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from ...core.labels import MultilabelLabeler
from ...core.args import CommonArguments
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
    lrp = 'lrp' in coll
    data_file_name = os.path.join(
        arg.data_in_dir, f'lrp_{coll}.csv' if lrp else f'{coll}.csv'
    )
    if not os.path.exists(data_file_name):
        data_file_name = os.path.join(
            arg.data_out_dir, f'lrp_{coll}.csv' if lrp else f'{coll}.csv'
        )
    data_df = load_add_corpus_part(data_file_name, 'label')
    return data_df.to_dict(orient='records')


def _chunk_data(data_list, chunk_size=500):
    """Generator function to yield data in chunks"""
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i + chunk_size]


def _prepare_documents(coll: str, models, data_chunk):
    """Prepare documents for bulk indexing"""
    lrp = 'lrp' in coll
    for data_item in data_chunk:
        # logger.info('Processing item [%s]', data_item)
        for model_name, model in models.items():
            data_item['m_' + model_name] = model(data_item['text'])[0].tolist()
        op = {
            '_op_type': 'index',
            '_index': f'lrp_{coll}' if lrp else f'{coll}',
            '_id': data_item['uuid'] if 'uuid' in data_item else data_item['a_uuid'],
            '_source': data_item
        }
        yield op


# noinspection DuplicatedCode
def es_pump(arg) -> int:
    """
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
    os.environ['HF_HOME'] = arg.tmp_dir  # local tmp dir

    compute_arg_collection_name(arg)
    tokenizers = {}
    for lang in arg.lang:
        tokenizers[lang] = get_segmenter(lang, arg.tmp_dir)
    bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def bge_m3_embed(text_to_embed: str):
        return bge_m3_model.encode([text_to_embed])['dense_vecs']

    models = {
        'bge_m3': bge_m3_embed
    }

    data_as_dicts = _load_data(arg, arg.collection)

    client = Elasticsearch(CLIENT_URL)
    try:
        if not client.indices.exists(index=arg.collection):
            logger.warning('Collection [%s] does not exist.', arg.collection)
            return 1

        total_indexed = 0
        total_failed = 0
        for chunk in _chunk_data(data_as_dicts, chunk_size=1000):
            success, failed = bulk(
                client=client,
                actions=_prepare_documents(arg.collection, models, chunk),
                request_timeout=60
            )

            total_indexed += success
            total_failed += len(failed) if failed else 0
            logger.info(f"Processed chunk: {success} succeeded, {len(failed) if failed else 0} failed")
    finally:
        client.close()

    return 0


# noinspection DuplicatedCode
def es_dedup(arg) -> int:
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
    os.environ['HF_HOME'] = arg.tmp_dir  # local tmp dir
    compute_arg_collection_name(arg)
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
            find_similar(client, arg.collection, data_item['a_uuid'], data_item['m_bge_m3'], on_similar)
            return True

        load_data(client, arg.collection, start_date, end_date, on_item)
    finally:
        client.close()

    lrp = 'lrp' in arg.collection
    data_file_name = os.path.join(
        arg.data_out_dir, f'lrp_{arg.collection}.csv' if lrp else f'{arg.collection}_duplicates.csv'
    )
    df = pd.DataFrame(duplicates)
    df.to_csv(data_file_name, index=False, encoding='utf8')
    return 0


# noinspection DuplicatedCode
def es_test_bge_m3(arg) -> int:
    """
    ./mulabel es test_bge_m3 -c lrp_mulabel
    ./mulabel es test_bge_m3 -c lrp_mulabel -l sl,sr

    ./mulabel es test_bge_m3 -c lrp_mulabel -l sl --public
    ./mulabel es test_bge_m3 -c lrp_mulabel -l sl --public

    ./mulabel es test_bge_m3 -c mulabel -l sl
    ./mulabel es test_bge_m3 -c mulabel -l sl --public
    ./mulabel es test_bge_m3 -c mulabel -l sl --public --seed_only
    """
    os.environ['HF_HOME'] = arg.tmp_dir  # local tmp dir

    compute_arg_collection_name(arg)
    tokenizers = {}
    for lang in arg.lang:
        tokenizers[lang] = get_segmenter(lang, arg.tmp_dir)
    bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def bge_m3_embed(text_to_embed: str):
        return bge_m3_model.encode([text_to_embed])['dense_vecs']

    models = {
        'bge_m3': bge_m3_embed
    }

    train_coll_name = arg.collection + '_train'
    test_coll_name = arg.collection + '_test'
    data_as_dicts = _load_data(arg, test_coll_name)

    labels_file_name = os.path.join(arg.data_in_dir, f'{arg.collection}_labels.csv')
    with open(labels_file_name, 'r') as l_file:
        all_labels = [line.split(',')[0].strip() for line in l_file]
    if all_labels[0] == 'label':
        all_labels.pop(0)
    labeler = MultilabelLabeler(all_labels)
    labeler.fit()

    state = {'doc': {}}
    y_true = []
    y_pred = []
    client = Elasticsearch(CLIENT_URL)
    try:
        if not client.indices.exists(index=train_coll_name):
            logger.warning('Collection [%s] does not exist.', train_coll_name)
            return 1

        def on_similar(data_item: Dict[str, Any], score: float) -> bool:
            y_pred.append(labeler.vectorize([data_item['label']])[0])
            y_true.append(labeler.vectorize([state['doc']["label"]])[0])
            return True

        for data_item in tqdm(data_as_dicts, desc='Processing zero shot complete eval.'):
            for model_name, model in models.items():
                data_item['m_' + model_name] = model(data_item['text'])[0].tolist()
            state['doc'] = data_item
            num_ret = find_similar(
                client, train_coll_name, data_item['a_uuid'], data_item['m_bge_m3'], on_similar
            )
            if num_ret == 0:
                y_pred.append(labeler.vectorize([[]])[0])

    finally:
        client.close()

    average_type = 'micro'
    p = precision_score(y_true, y_pred, average=average_type)
    r = recall_score(y_true, y_pred, average=average_type)
    f1 = f1_score(y_true, y_pred, average=average_type)
    print(f'Precision:{p}\nRecall:{r}\nF1:{f1}')

    return 0
