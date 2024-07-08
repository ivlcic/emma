import json
import os
import logging
import weaviate

from argparse import ArgumentParser
from typing import Callable, Dict, Any, List
from weaviate import WeaviateClient
from weaviate.collections.classes.config import Configure, Property, DataType, VectorDistances

from ..tokenizer import get_tokenizer
from ..utils import load_map_file
from ...core.args import CommonArguments

logger = logging.getLogger('mulabel.db')

__supported_languages = [
    'sl', 'sr', 'sq', 'mk', 'bs', 'hr', 'bg', 'en', 'uk', 'ru',
    'sk', 'cs', 'ro', 'hu', 'pl', 'pt', 'el', 'de', 'es', 'it'
]


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


def _process_arg_lang(arg):
    arg.collection_conf = arg.collection
    if arg.lang:
        arg.lang = arg.lang.split(',')
        not_in_languages = [lang for lang in arg.lang if lang not in __supported_languages]
        if not_in_languages:
            logger.error('Languages %s are not supported!', [])
        arg.collection = arg.collection + '_' + '_'.join(arg.lang)
    else:
        arg.lang = __supported_languages


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
    ./mulabel db init -c Label -l sl,sr
    """
    _process_arg_lang(arg)
    client = weaviate.connect_to_local()
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
    _process_arg_lang(arg)
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


def _get_sentences_at(c_idx: int, num_sent: int, segments: Dict[str, List[Any]]):
    result = []
    s_idx = 0
    for i, char_idx in enumerate(segments['indices']):
        s_idx = i - 1 if i > 0 else 0
        if char_idx <= c_idx:
            continue
        break

    if s_idx < 0:
        return result
    offset = (num_sent - 1) // 2
    for i in range(s_idx - offset, s_idx + offset):
        if i < 0:
            continue
        if i > len(segments['sentences']) - 1:
            break
        result.append(segments['sentences'][i])
    if not result:
        return [segments['sentences'][s_idx]]
    return result


def _sentence_segment(text: str, tokenize: Callable, segments: Dict[str, List[Any]]) -> None:
    doc = tokenize(text)
    last_offset = 0
    for i, sentence in enumerate([sentence.text for sentence in doc.sentences]):
        segments['sentences'].append(sentence)
        segments['indices'].append(text.index(sentence))
        last_offset += len(sentence)


def db_pump(arg) -> int:
    """
        ./mulabel db pump -c Label
        ./mulabel db pump -c Label -l sl,sr
        """
    _process_arg_lang(arg)

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

    tokenizers = {}
    for lang in arg.lang:
        tokenizers[lang] = get_tokenizer(lang, arg.data_in_dir)

    client = weaviate.connect_to_local()

    try:
        if not client.collections.exists(arg.collection):
            logger.warning('Collection [%s] does not exist.', arg.collection)
            return 1
        for article_idx, article in enumerate(json_data):
            if article['lang'] not in arg.lang:
                continue
            tokenize = tokenizers[article['lang']]
            if not tokenize:
                logger.warning('Missing [%s] tokenizer', article['lang'])
                continue
            span_fields = {
                'ts': {'name': 'title', 'sentences': [], 'indices': []},
                'bs': {'name': 'body', 'sentences': [], 'indices': []},
            }
            prev_text = ''
            text = ''
            for seg_name, span_field in span_fields.items():
                if 'text' in article[span_field['name']]:
                    text = article[span_field['name']]['text']
                if text:
                    _sentence_segment(text, tokenize, span_field)

                # remove title from body if body starts wit a title and prepend it as a sentence
                if prev_text and text.startswith(prev_text):
                    text = text[len(prev_text):]
                    text = prev_text + '.\n\n' + text
                prev_text = text

            for tag in article['tags']:
                span_names = span_fields.keys()
                all_spans_empty = all(not tag[key] for key in span_names)
                if all_spans_empty:
                    logger.info('Article [%s] label [%s] has no spans', article_idx, tag['id'])
                    continue

                spans = {}
                for span_name in span_names:
                    for span in tag[span_name]:
                        if span['kwe'] not in spans:
                            spans[span['kwe']] = {
                                'm': [],
                                's': []
                            }
                        logger.debug(
                            'Getting relevant [%s] sentences at [%s] for [%s]',
                            span['kwe'], span['s'], span_name
                        )
                        relevant_sentences = _get_sentences_at(span['s'], 1, span_fields[span_name])
                        spans[span['kwe']]['m'].append(span['m'])
                        spans[span['kwe']]['s'].extend(relevant_sentences)
                logger.info(
                    'Article [%s] label [%s] has relevant sentences %s',
                    article_idx, tag['id'], spans
                )

            if article_idx > 1:
                return 1

    finally:
        client.close()
    return 0


def prep_analyse(arg) -> int:
    pass
