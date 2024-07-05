import json
import os
import logging
import weaviate

from argparse import ArgumentParser
from typing import Callable, Dict, Any
from weaviate import WeaviateClient
from weaviate.collections.classes.config import Configure, Property, DataType, VectorDistances

from ..tokenizer import get_tokenizer
from ..utils import load_map_file
from ...core.args import CommonArguments

logger = logging.getLogger('mulabel.db')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.tmp_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_out_dir'))
    parser.add_argument(
        '-c', '--collection', help='Collection to manage.', type=str
    )


def _get_collection_conf(coll_name: str) -> Dict[str, Any]:
    conf = {
        'Label': {
            'properties': [
                Property(
                    name='uuid', description='Sentence ID.', data_type=DataType.UUID,
                    index_searchable=False, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='a_uuid', description='Article ID.', data_type=DataType.UUID,
                    index_searchable=False, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='country', description='Article Country.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='language', description='Article Language.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='m_id', description='Article Media ID.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='label_id', description='Article Matched Sentence Label ID.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='label_title', description='Article Matched Sentence Label Title.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='kwe_id', description='Article Matched Keyword Expression ID.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='kwe', description='Article Matched Keyword Expression.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='kwe_s', description='Keyword Expression Start Index.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='content', description='The article content.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
            ],
            'vectorizer_config': [
                Configure.NamedVectors.none(
                    name="vanilla_bge_m3", vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE, vector_cache_max_objects=100000
                    )
                ),
            ]
        }
    }
    return conf[coll_name]


def _create_collection_if_not_exists(
        client: WeaviateClient,
        collection_name: str,
        collection_conf_provider: Callable[[str], Dict[str, Any]]) -> None:
    # Check if the collection exists
    if not client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' does not exist. Creating...")
        logger.info('Collection [%s] created successfully.', collection_name)
        client.collections.create(**collection_conf_provider(collection_name))
        logger.info('Collection [%s] created successfully.', collection_name)
    else:
        logger.warning('Collection [%s] already exists.', collection_name)


def db_init(arg) -> int:
    """
    ./mulabel db init -c Label
    """
    client = weaviate.connect_to_local()
    try:
        coll_conf = _get_collection_conf(arg.collection)
        # create collection if it doesn't exist
        if not client.collections.exists(arg.collection):
            logger.info('Collection [%s] does not exist. Creating...', arg.collection)
            client.collections.create(arg.collection, **coll_conf)
            logger.info('Collection [%s] created successfully.', arg.collection)
        else:
            logger.warning('Collection [%s] already exists.', arg.collection)
    finally:
        client.close()
    return 0


def db_drop(arg) -> int:
    """
    ./mulabel db drop -c Label
    """
    client = weaviate.connect_to_local()
    try:
        if client.collections.exists(arg.collection):
            client.collections.delete(arg.collection)
            logger.info('Collection [%s] deleted.', arg.collection)
        else:
            logger.warning('Collection [%s] does not exist.', arg.collection)
    finally:
        client.close()
    return 0


def db_pump(arg) -> int:
    kwes = load_map_file(
        os.path.join(arg.data_out_dir, 'map_kwe_tags.csv'), ['tag_id', 'expr']
    )

    labels = load_map_file(
        os.path.join(arg.data_out_dir, 'map_tags.csv'),
        ['name', 'count', 'parent_id', 'monitoring_country', 'monitoring_industry']
    )

    file_name = os.path.join(arg.data_out_dir, 'data_2023_01.jsonl')
    with open(file_name, 'r', encoding='utf8') as json_file:
        json_data = json.load(json_file)

    tokenizers = {
        'sl': get_tokenizer('sl', arg.data_in_dir),
        'sr': get_tokenizer('sr', arg.data_in_dir),
        'sq': get_tokenizer('sr', arg.data_in_dir),
        'mk': get_tokenizer('mk', arg.data_in_dir),
        'bs': get_tokenizer('hr', arg.data_in_dir),
        'hr': get_tokenizer('hr', arg.data_in_dir),
        'bg': get_tokenizer('bg', arg.data_in_dir),
        'en': get_tokenizer('en', arg.data_in_dir),
        'uk': get_tokenizer('uk', arg.data_in_dir),
        'ru': get_tokenizer('ru', arg.data_in_dir),
        'sk': get_tokenizer('sk', arg.data_in_dir),
        'cs': get_tokenizer('cs', arg.data_in_dir),
        'ro': get_tokenizer('ro', arg.data_in_dir),
        'hu': get_tokenizer('hu', arg.data_in_dir),
        'pl': get_tokenizer('pl', arg.data_in_dir),
        'pt': get_tokenizer('pt', arg.data_in_dir),
        'el': get_tokenizer('el', arg.data_in_dir),
        'de': get_tokenizer('de', arg.data_in_dir),
        'es': get_tokenizer('es', arg.data_in_dir),
        'it': get_tokenizer('it', arg.data_in_dir)
    }

    client = weaviate.connect_to_local()
    try:
        if not client.collections.exists(arg.collection):
            logger.warning('Collection [%s] does not exist.', arg.collection)
            return 1
        for a in json_data:
            if 'zh' == a['lang']:
                continue
            tokenizer = tokenizers[a['lang']]
            tokenizer.tokenize(a['title'])
            tokenizer.tokenize(a['body'])

    finally:
        client.close()
    return 0


def prep_analyse(arg) -> int:
    pass
