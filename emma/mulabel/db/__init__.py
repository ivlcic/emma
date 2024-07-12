import json
import os
import logging

import weaviate
from weaviate.util import generate_uuid5

logger = logging.getLogger('mulabel.db')
os.environ['HF_HOME'] = os.path.abspath(os.path.join(__file__, '..', '..', '..', '..', 'tmp', 'mulabel'))
logger.info('Setting [HF_HOME] to [%s]', os.environ["HF_HOME"])

from argparse import ArgumentParser
from typing import Callable, Dict, Any, List
from weaviate import WeaviateClient
from weaviate.collections.classes.config import Configure, Property, DataType, VectorDistances
from FlagEmbedding import BGEM3FlagModel

from ..tokenizer import get_tokenizer
from ..utils import load_map_file
from ...core.args import CommonArguments


__supported_languages = [
    'sl', 'sr', 'sq', 'mk', 'bs', 'hr', 'bg', 'en', 'uk', 'ru',
    'sk', 'cs', 'ro', 'hu', 'pl', 'pt', 'el', 'de', 'es', 'it'
]

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
                    name='a_uuid', description='Article ID.', data_type=DataType.UUID,
                    index_searchable=False, index_filterable=True, skip_vectorization=True
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
                    name="bge_m3", vector_index_config=Configure.VectorIndex.hnsw(
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
    _process_arg_lang(arg)
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


def _get_sentence_index_at(c_idx: int, segments: Dict[str, List[Any]]) -> int:
    s_idx = -1
    last_idx = len(segments['indices']) - 1
    for i, s_char_idx in enumerate(segments['indices']):
        e_char_idx = segments['indices'][i + 1] if i + 1 <= last_idx else s_char_idx
        if s_char_idx <= c_idx < e_char_idx:
            s_idx = i
            break

    return s_idx


def _sentence_segment(text: str, tokenize: Callable, segments: Dict[str, List[Any]]) -> None:
    doc = tokenize(text)
    for i, sentence in enumerate([sentence.text for sentence in doc.sentences]):
        segments['sentences'].append(sentence)
        segments['indices'].append(text.index(sentence))
    segments['indices'].append(segments['indices'][-1] + len(sentence))


def _construct_db_labels(article: Dict[str, Any], segment_spans: Dict[str, Any],
                         maps: Dict[str, Any]) -> List[Dict[str, Any]]:
    segment_spans = {
        'ts': {'name': 'title', 'sentences': [], 'indices': []},
        'bs': {'name': 'body', 'sentences': [], 'indices': []},
    }
    text = ''
    for seg_name, span_field in segment_spans.items():
        if 'text' in article[span_field['name']]:
            text = article[span_field['name']]['text']
        if text:
            _sentence_segment(text, tokenize, span_field)

    t_sent = segment_spans['ts']['sentences']
    b_sent = segment_spans['bs']['sentences']
    text = ''
    if len(t_sent) > 0 and len(b_sent) > 0:
        b_first = b_sent[0].replace(' ', '').lower()
        t_all = ''.join(t_sent).replace(' ', '').lower()
        if b_first.startswith(t_all):
            text = ' '.join(b_sent)
    if not text:
        text = ' '.join(t_sent) + '\n\n' + ' '.join(b_sent)

    db_labels = []
    seed_label = {
        'id': article['id'],
        'm_id': article['m_id'],
        'country': article['country'],
        'lang': article['lang'],
        'passage_cat': 0,
        'passage': text,
        'kwe': [],
        'kwe_id': []
    }

    label_ids_sent_idx = {}
    all_sentences = []
    for tag in article['tags']:
        segment_names = segment_spans.keys()
        all_spans_empty = all(not tag[key] for key in segment_names)
        db_label = seed_label.copy()
        db_label['label_id'] = tag['id']
        db_label['label_title'] = maps['labels'][tag['id']]['name']
        db_label['kwe'] = []
        db_label['kwe_id'] = []
        if all_spans_empty:
            logger.info(
                'Article [%s] label [%s::%s] has no spans',
                article['id'], tag['id'], maps['labels'][tag['id']]['name']
            )
            db_labels.append(db_label)
            continue

        if tag['id'] not in label_ids_sent_idx:
            label_ids_sent_idx[tag['id']] = {'s_idx': [], 'kwe': [], 'kwe_id': [], 'title': ''}

        prev_segment_offset = 0
        for segment_name in segment_names:
            sentences = segment_spans[segment_name]['sentences']
            for span in tag[segment_name]:
                # single sentence passage matching keyword expression (1 label <-> N kwe)
                db_span_label = db_label.copy()
                db_span_label['kwe_id'] = [span['kwe']]
                db_span_label['kwe'] = [span['m']]
                center_sentence_idx = _get_sentence_index_at(span['s'], segment_spans[segment_name])
                if 0 <= center_sentence_idx < len(sentences):
                    sentence = segment_spans[segment_name]['sentences'][center_sentence_idx]
                    label_ids_sent_idx[tag['id']]['s_idx'].append(center_sentence_idx + prev_segment_offset)
                    label_ids_sent_idx[tag['id']]['kwe'].append(span['m'])
                    label_ids_sent_idx[tag['id']]['kwe_id'].append(span['kwe'])
                    label_ids_sent_idx[tag['id']]['title'] = db_label['label_title']
                    db_span_label_1 = db_span_label.copy()
                    db_span_label_1['passage_cat'] = 1
                    db_span_label_1['passage'] = sentence
                    db_labels.append(db_span_label_1)
            all_sentences.extend(sentences)
            prev_segment_offset += len(sentences)

    # add a larger (multi-sentence) passage matching keyword expression
    # considering overlaps also ... hence the complicated code
    passage_sizes = [3, 5]  # we consider 3 and 5 sentence contexts
    for passage_size in passage_sizes:
        for label_name, label_data in label_ids_sent_idx.items():
            # store all sentence indices for a given keyword expression match
            # remove the ones that were included in a passage to minimize redundant overlapping
            passage_center_sentence_indices = set(label_data['s_idx'])
            for center_sentence_idx in label_data['s_idx']:
                if center_sentence_idx not in passage_center_sentence_indices:
                    continue
                sent_offset = passage_size // 2
                start = center_sentence_idx - sent_offset
                if start < 0:
                    start = 0
                end = start + passage_size
                if end > len(all_sentences) - 1:
                    end = len(all_sentences) - 1
                passage = []
                for i in range(start, end):
                    if i in passage_center_sentence_indices:
                        passage_center_sentence_indices.remove(i)
                    passage.append(all_sentences[i])
                db_span_label = seed_label.copy()
                db_span_label['kwe_id'].extend(label_data['kwe_id'])
                db_span_label['kwe'].extend(label_data['kwe'])
                db_span_label['label_id'] = label_name
                db_span_label['label_title'] = label_data['title']
                db_span_label['passage_cat'] = passage_size
                db_span_label['passage'] = ' '.join(passage)
                db_labels.append(db_span_label)
                pass

    return db_labels


def db_pump(arg) -> int:
    """
        ./mulabel db pump -c Label
        ./mulabel db pump -c Label -l sl,sr
        """
    _process_arg_lang(arg)
    maps = {
        'kwes': load_map_file(
            os.path.join(arg.data_out_dir, 'map_kwe_tags.csv'), ['tag_id', 'expr']
        ),
        'labels': load_map_file(
            os.path.join(arg.data_out_dir, 'map_tags.csv'),
            ['name', 'count', 'parent_id', 'monitoring_country', 'monitoring_industry']
        )
    }

    file_name = os.path.join(arg.data_out_dir, 'data_2023_01.jsonl')
    with open(file_name, 'r', encoding='utf8') as json_file:
        json_data = json.load(json_file)

    tokenizers = {}
    for lang in arg.lang:
        tokenizers[lang] = get_tokenizer(lang, arg.data_in_dir)

    bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def bge_m3_embed(text_to_embed: str):
        return bge_m3_model.encode([text_to_embed])['dense_vecs']

    models = {
        'bge_m3': bge_m3_embed
    }

    client = weaviate.connect_to_local(port=__WEAVIATE_PORT)

    try:
        if not client.collections.exists(arg.collection):
            logger.warning('Collection [%s] does not exist.', arg.collection)
            return 1
        coll = client.collections.get(arg.collection)
        for article_idx, article in enumerate(json_data):
            if article['lang'] not in arg.lang:
                continue
            tokenize = tokenizers[article['lang']]
            if not tokenize:
                logger.warning('Missing [%s] tokenizer', article['lang'])
                continue
            segment_spans = {
                'ts': {'name': 'title', 'sentences': [], 'indices': []},
                'bs': {'name': 'body', 'sentences': [], 'indices': []},
            }
            text = ''
            for seg_name, span_field in segment_spans.items():
                if 'text' in article[span_field['name']]:
                    text = article[span_field['name']]['text']
                if text:
                    _sentence_segment(text, tokenize, span_field)

            db_labels = _construct_db_labels(article, segment_spans, maps)
            for db_label in db_labels:
                vectors = {}
                for model_name, model in models.items():
                    vectors['m_' + model_name] = model(db_label['passage'])
                coll.data.insert(
                    uuid=generate_uuid5(db_label),
                    properties=db_label,
                    vector=vectors
                )
            if article_idx > 1:
                return 1

    finally:
        client.close()
    return 0


def prep_analyse(arg) -> int:
    return 0
