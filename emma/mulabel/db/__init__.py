import csv
import json
import os
import logging
import weaviate

from argparse import ArgumentParser
from typing import Callable, Dict, Any
from weaviate import WeaviateClient
from weaviate.util import generate_uuid5
from weaviate.collections.classes.config import Configure, Property, DataType, VectorDistances
from FlagEmbedding import BGEM3FlagModel

from ..tokenizer import get_segmenter
from ..utils import load_map_file, construct_span_contexts, write_map_file, __supported_languages, \
    compute_arg_collection_name
from ...core.args import CommonArguments

logger = logging.getLogger('mulabel.utils')


__WEAVIATE_PORT = 18484


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.tmp_dir(module_name, parser, ('-i', '--data_in_dir'))
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


def _get_collection_conf(coll_name: str) -> Dict[str, Any]:
    conf = {
        'Label': {
            'properties': [
                Property(
                    name='a_uuid', description='Article UUID.', data_type=DataType.UUID,
                    index_searchable=False, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='a_id', description='Article ID.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='country', description='Article Country.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='lang', description='Article Language.', data_type=DataType.TEXT,
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
                    name='kwe_id', description='Article Matched Keyword Expression ID.', data_type=DataType.TEXT_ARRAY,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='kwe', description='Article Matched Keyword Expression.', data_type=DataType.TEXT_ARRAY,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='passage_cat', description='Passage size category (number of sentences)',
                    data_type=DataType.INT, index_searchable=False, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='passage', description='The article passage content.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
            ],
            'vectorizer_config': [
                Configure.NamedVectors.none(
                    name="m_bge_m3", vector_index_config=Configure.VectorIndex.hnsw(
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
    ./mulabel db init -c Label -l sl
    ./mulabel db init -c Label -l sl,sr
    """
    compute_arg_collection_name(arg)
    client = weaviate.connect_to_local(port=__WEAVIATE_PORT)
    try:
        if arg.collection_conf:
            coll_conf = _get_collection_conf(arg.collection_conf)
        else:
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
    ./mulabel db drop -c Label -l sl,sr
    """
    compute_arg_collection_name(arg)
    client = weaviate.connect_to_local(port=__WEAVIATE_PORT)
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
    """
        ./mulabel db pump -c Label
        ./mulabel db pump -c Label -l sl,sr

        ./mulabel db drop -c Label -l sl
        ./mulabel db init -c Label -l sl
        ./mulabel db pump -c Label -l sl --public

        ./mulabel db pump -c Label -l sl --public --seed_only
        """
    compute_arg_collection_name(arg)
    label_cols = ['name', 'count', 'parent_id', 'monitoring_country', 'monitoring_industry']
    maps = {
        'kwes': load_map_file(
            os.path.join(arg.data_out_dir, 'map_kwe_tags.csv'), ['tag_id', 'expr']
        ),
        'labels': load_map_file(
            os.path.join(arg.data_out_dir, 'map_tags.csv'), label_cols
        ),
        'seed_labels': load_map_file(
            os.path.join(arg.data_out_dir, 'map_seed_tags.csv'), label_cols
        ),
        'trained_labels': {}
    }
    # postfix = '2023_99'
    max_articles = -1
    postfix = '2023_01'
    file_name = os.path.join(arg.data_out_dir, f'data_{postfix}.jsonl')
    with open(file_name, 'r', encoding='utf8') as json_file:
        json_data = json.load(json_file)

    article_map_file_name = os.path.join(arg.data_out_dir, f'map_articles_{postfix}.csv')
    article_cols = [
        'uuid', 'public', 'created', 'published', 'country', 'mon_country', 'lang', 'script', 'm_id',
        'rel_path', 'url', 'sent', 'words', 'sp_tokens', 'tags_count', 'tags'
    ]
    map_articles = load_map_file(article_map_file_name, article_cols)

    tokenizers = {}
    for lang in arg.lang:
        tokenizers[lang] = get_segmenter(lang, arg.data_in_dir)

    bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def bge_m3_embed(text_to_embed: str):
        return bge_m3_model.encode([text_to_embed])['dense_vecs']

    models = {
        'bge_m3': bge_m3_embed
    }

    db_file_name = os.path.join(arg.data_in_dir, f'db_file_{postfix}.csv')
    with open(db_file_name, 'w', encoding='utf8') as db_file:
        writer = csv.writer(db_file)

        header_written = False
        client = weaviate.connect_to_local(port=__WEAVIATE_PORT)

        try:
            if not client.collections.exists(arg.collection):
                logger.warning('Collection [%s] does not exist.', arg.collection)
                return 1
            coll = client.collections.get(arg.collection)
            stored = set()
            for article_idx, article in enumerate(json_data):
                if article['lang'] not in arg.lang:
                    continue
                if arg.public and article['public'] != 1:
                    continue
                if arg.seed_only:
                    article['tags'] = [x for x in article['tags'] if x['id'] in maps['seed_labels']]

                if article['id'] in map_articles:
                    article['uuid'] = map_articles[article['id']]['uuid']
                else:
                    logger.warning('Missing mapped article [%s]', article['id'])

                tokenize = tokenizers[article['lang']]
                if not tokenize:
                    logger.warning('Missing [%s] tokenizer', article['lang'])
                    continue

                for tag in article['tags']:
                    if not tag['id'] in maps['trained_labels']:
                        maps['trained_labels'][tag['id']] = maps['labels'][tag['id']]

                db_labels = construct_span_contexts(
                    article, tokenize, maps, [1, 3, 5, 7, 9]
                )
                for db_label in db_labels:
                    record_uuid = generate_uuid5(db_label)
                    if record_uuid in stored:
                        continue

                    logger.info('Processing label [%s]', db_label)

                    vectors = {}
                    for model_name, model in models.items():
                        vectors['m_' + model_name] = model(db_label['passage'])

                    if not header_written:
                        cols = ['uuid']
                        cols.extend(db_label.keys())
                        writer.writerow(cols)
                        header_written = True
                    cols = [record_uuid]
                    cols.extend(db_label.values())
                    writer.writerow(cols)

                    coll.data.insert(
                        uuid=record_uuid,
                        properties=db_label,
                        vector=vectors
                    )
                    stored.add(record_uuid)
                if article_idx < max_articles:
                    break
        finally:
            client.close()

    write_map_file(maps['trained_labels'], os.path.join(arg.data_out_dir, 'map_trained_tags.csv'), label_cols)
    return 0


def prep_analyse(arg) -> int:
    return 0
