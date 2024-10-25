import ast
import os
import logging
import pandas as pd
import pandas.api.types as ptypes

from argparse import ArgumentParser
from typing import Callable, Dict, Any

from elasticsearch import Elasticsearch
from weaviate.util import generate_uuid5
from FlagEmbedding import BGEM3FlagModel

from ...core.labels import MultilabelLabeler
from ..tokenizer import get_segmenter
from ..utils import __supported_languages, compute_arg_collection_name
from ...core.args import CommonArguments

logger = logging.getLogger('mulabel.es')

CLIENT_URL = "http://localhost:9266/"


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_out_dir'))
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
    ./mulabel db init -c lrp
    ./mulabel db init -c lrp -l sl
    ./mulabel db init -c lrp -l sl,sr
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
    ./mulabel es drop -c lrp
    ./mulabel es drop -c lrp -l sl,sr
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


def es_pump(arg) -> int:
    """
    ./mulabel es pump -c lrp
    ./mulabel es pump -c lrp -l sl,sr

    ./mulabel es drop -c lrp -l sl --public
    ./mulabel es init -c lrp -l sl --public
    ./mulabel es pump -c lrp -l sl --public
    ./mulabel es pump -c complete -l sl --public

    ./mulabel es pump -c lrp -l sl --public --seed_only
    """
    lpr = 'lrp' in arg.collection

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

    if lpr:
        lrp_file_name = os.path.join(arg.data_in_dir, f'{arg.collection}_filtered_lrp_train.csv')
        lrp_df = pd.read_csv(lrp_file_name)
        if ptypes.is_string_dtype(lrp_df['kwe_id']):
            lrp_df['kwe_id'] = lrp_df['kwe_id'].apply(ast.literal_eval)
        if ptypes.is_string_dtype(lrp_df['kwe']):
            lrp_df['kwe'] = lrp_df['kwe'].apply(ast.literal_eval)
        lrp_dicts = lrp_df.to_dict(orient='records')
    else:
        lrp_file_name = os.path.join(arg.data_in_dir, f'{arg.collection}_filtered_article_train.csv')
        lrp_df = pd.read_csv(lrp_file_name)
        lrp_df.rename(columns={'id': 'a_id'}, inplace=True)
        if ptypes.is_string_dtype(lrp_df['labels']):
            lrp_df['labels'] = lrp_df['labels'].apply(ast.literal_eval)
        lrp_dicts = lrp_df.to_dict(orient='records')

    client = Elasticsearch(CLIENT_URL)
    try:
        if not client.indices.exists(index=arg.collection):
            logger.warning('Collection [%s] does not exist.', arg.collection)
            return 1

        stored = set()
        for lrp_dict in lrp_dicts:
            record_uuid = generate_uuid5(lrp_dict)
            if record_uuid in stored:
                continue

            # this is a for the lrp case
            if lpr:
                if 'kwe_id' in lrp_dict:
                    lrp_kwes = []
                    for i, kwe_id in lrp_dict['kwe_id']:
                        lrp_kwe = {'id': kwe_id, 'value': lrp_dict['kwe'][i]}
                        lrp_kwes.append(lrp_kwe)
                    del lrp_dict['kwe_id']
                    lrp_dict['kwe'] = lrp_kwes
                if 'label_id' in lrp_dict:
                    lrp_kwes = []
                    for i, kwe_id in lrp_dict['label_id']:
                        lrp_kwe = {'id': kwe_id, 'title': lrp_dict['label'][i]}
                        lrp_kwes.append(lrp_kwe)
                    del lrp_dict['label_id']
                    lrp_dict['label'] = lrp_kwes

            logger.info('Processing label [%s]', lrp_dict)
            for model_name, model in models.items():
                if lpr:
                    lrp_dict['m_' + model_name] = model(lrp_dict['passage'])[0].tolist()
                else:
                    lrp_dict['m_' + model_name] = model(lrp_dict['text'])[0].tolist()
            client.index(
                index=arg.collection,
                id=record_uuid,
                document=lrp_dict,
            )
            stored.add(record_uuid)
    finally:
        client.close()

    return 0
