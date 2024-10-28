import ast
import os
import logging
import pandas as pd
import pandas.api.types as ptypes

from argparse import ArgumentParser
from typing import Callable, Dict, Any, List

from elasticsearch import Elasticsearch
from weaviate.util import generate_uuid5
from FlagEmbedding import BGEM3FlagModel

from ..tokenizer import get_segmenter
from ..utils import __supported_languages, compute_arg_collection_name
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


def _load_data(arg, lrp: bool) -> List[Dict[str, Any]]:
    if lrp:
        data_file_name = os.path.join(arg.data_in_dir, f'{arg.collection}_filtered_lrp_train.csv')
        data_frame = pd.read_csv(data_file_name)
        if ptypes.is_string_dtype(data_frame['kwe_id']):
            data_frame['kwe_id'] = data_frame['kwe_id'].apply(ast.literal_eval)
        if ptypes.is_string_dtype(data_frame['kwe']):
            data_frame['kwe'] = data_frame['kwe'].apply(ast.literal_eval)
        data_as_dicts = data_frame.to_dict(orient='records')
    else:
        data_file_name = os.path.join(arg.data_in_dir, f'{arg.collection}_filtered_article_train.csv')
        data_frame = pd.read_csv(data_file_name)
        data_frame.rename(columns={'id': 'a_id'}, inplace=True)
        if ptypes.is_string_dtype(data_frame['labels']):
            data_frame['labels'] = data_frame['labels'].apply(ast.literal_eval)
        data_as_dicts = data_frame.to_dict(orient='records')
    return data_as_dicts


def _restruct(lrp, data_item) -> None:
    if not lrp:
        data_item['id'] = data_item['a_uuid']
        return
    data_item['id'] = data_item['uuid']
    # this is a for the lrp case
    if 'kwe_id' in data_item:
        lrp_kwes = []
        for i, kwe_id in data_item['kwe_id']:
            lrp_kwe = {'id': kwe_id, 'value': data_item['kwe'][i]}
            lrp_kwes.append(lrp_kwe)
        del data_item['kwe_id']
        data_item['kwe'] = lrp_kwes
    if 'label_id' in data_item:
        lrp_kwes = []
        for i, kwe_id in data_item['label_id']:
            lrp_kwe = {'id': kwe_id, 'title': data_item['label'][i]}
            lrp_kwes.append(lrp_kwe)
        del data_item['label_id']
        data_item['label'] = lrp_kwes


def es_pump(arg) -> int:
    """
    ./mulabel es pump -c lrp_mulabel
    ./mulabel es pump -c lrp_mulabel -l sl,sr

    ./mulabel es drop -c lrp_mulabel -l sl --public
    ./mulabel es init -c lrp_mulabel -l sl --public
    ./mulabel es pump -c lrp_mulabel -l sl --public

    ./mulabel es pump -c mulabel -l sl --public

    ./mulabel es pump -c mulabel -l sl --public --seed_only
    """
    os.environ['HF_HOME'] = arg.tmp_dir  # local tmp dir

    lrp = 'lrp' in arg.collection
    compute_arg_collection_name(arg)
    tokenizers = {}
    for lang in arg.lang:
        tokenizers[lang] = get_segmenter(lang, arg.data_in_dir)

    bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def bge_m3_embed(text_to_embed: str):
        return bge_m3_model.encode([text_to_embed])['dense_vecs']

    models = {
        'bge_m3': bge_m3_embed
    }

    data_as_dicts = _load_data(arg, lrp)
    client = Elasticsearch(CLIENT_URL)
    try:
        if not client.indices.exists(index=arg.collection):
            logger.warning('Collection [%s] does not exist.', arg.collection)
            return 1

        stored = set()
        for data_item in data_as_dicts:
            _restruct(lrp, data_item)

            logger.info('Processing item [%s]', data_item)
            for model_name, model in models.items():
                if lrp:
                    data_item['m_' + model_name] = model(data_item['passage'])[0].tolist()
                else:
                    data_item['m_' + model_name] = model(data_item['text'])[0].tolist()
            client.index(
                index=arg.collection,
                id=data_item['id'],
                document=data_item,
            )
            stored.add(data_item['id'])
    finally:
        client.close()

    return 0


def es_dedup(arg) -> int:
    """
    ./mulabel es pump -c mulabel_lrp
    ./mulabel es pump -c mulabel_lrp -l sl,sr

    ./mulabel es drop -c lrp_mulabel -l sl --public
    ./mulabel es init -c lrp_mulabel -l sl --public
    ./mulabel es pump -c lrp_mulabel -l sl --public

    ./mulabel es drop -c mulabel -l sl --public
    ./mulabel es init -c mulabel -l sl --public
    ./mulabel es pump -c mulabel -l sl --public
    ./mulabel es dedup -c mulabel -l sl --public
    """

    lrp = 'lrp' in arg.collection
    compute_arg_collection_name(arg)

    data_as_dicts = _load_data(arg, lrp)
    client = Elasticsearch(CLIENT_URL)
    try:
        stored = set()
        for data_item in data_as_dicts:
            record_uuid = generate_uuid5(data_item)
            if record_uuid in stored:
                continue

            _restruct(lrp, data_item)

            client.index(
                index=arg.collection,
                id=record_uuid,
                document=data_item,
            )
            stored.add(record_uuid)
    finally:
        client.close()

    return 0
