import ast
import os
import logging
import pandas as pd
import pandas.api.types as ptypes

from argparse import ArgumentParser
from typing import Callable, Dict, Any, List

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from FlagEmbedding import BGEM3FlagModel

from ..tokenizer import get_segmenter
from ..utils import __supported_languages, compute_arg_collection_name, load_add_corpus_part
from ...core.args import CommonArguments

logger = logging.getLogger('mulabel.es')

CLIENT_URL = "http://localhost:9266/"


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '-c', '--collection', help='Collection to manage.', type=str,
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
        type=int, default=1, choices=[1,3,5,7,9,0]
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


def _load_data(arg) -> List[Dict[str, Any]]:
    lrp = 'lrp' in arg.collection
    data_file_name = os.path.join(
        arg.data_in_dir, f'lrp_{arg.collection}.csv' if lrp else f'{arg.collection}.csv'
    )
    if not os.path.exists(data_file_name):
        data_file_name = os.path.join(
            arg.data_out_dir, f'lrp_{arg.collection}.csv' if lrp else f'{arg.collection}.csv'
        )
    data_df = load_add_corpus_part(data_file_name, 'label')
    return data_df.to_dict(orient='records')


def _chunk_data(data_list, chunk_size=500):
    """Generator function to yield data in chunks"""
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i + chunk_size]


def _prepare_documents(arg, models, data_chunk):
    """Prepare documents for bulk indexing"""
    lrp = 'lrp' in arg.collection
    for data_item in data_chunk:
        # logger.info('Processing item [%s]', data_item)
        for model_name, model in models.items():
            data_item['m_' + model_name] = model(data_item['text'])[0].tolist()
        op = {
            '_op_type': 'index',
            '_index': f'lrp_{arg.collection}' if lrp else f'{arg.collection}',
            '_id': data_item['uuid'] if 'uuid' in data_item else data_item['a_uuid'],
            'doc': data_item
        }
        yield op


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

    data_as_dicts = _load_data(arg)

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
                actions=_prepare_documents(arg, models, chunk),
                request_timeout=60
            )

            total_indexed += success
            total_failed += len(failed) if failed else 0
            print(f"Processed chunk: {success} succeeded, {len(failed) if failed else 0} failed")
    finally:
        client.close()

    return 0


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
    data_as_dicts = data_as_dicts = _load_data(arg)

    client = Elasticsearch(CLIENT_URL)
    try:
        for data_item in data_as_dicts:
            id = data_item['uuid'] if 'uuid' in data_item else data_item['a_uuid']
            client.index(
                index=arg.collection,
                id=id,
                document=data_item,
            )
    finally:
        client.close()

    return 0
