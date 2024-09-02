import ast
import os
import logging
import weaviate
import pandas as pd
import pandas.api.types as ptypes

from argparse import ArgumentParser
from typing import Callable, Dict, Any

from sklearn.metrics import precision_score, recall_score, f1_score
from weaviate import WeaviateClient
from weaviate.collections.classes.filters import Filter
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.util import generate_uuid5
from weaviate.collections.classes.config import Configure, Property, DataType, VectorDistances
from FlagEmbedding import BGEM3FlagModel

from ...core.labels import MultilabelLabeler
from ..tokenizer import get_segmenter
from ..utils import __supported_languages, compute_arg_collection_name
from ...core.args import CommonArguments

logger = logging.getLogger('mulabel.utils')


__WEAVIATE_PORT = 18484


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


def _get_collection_conf(coll_name: str) -> Dict[str, Any]:
    conf = {
        'mulabel': {
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
        },
        #id,lang,text,labels
        'whole_mulabel': {
            'properties': [
                Property(
                    name='a_id', description='Article ID.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='lang', description='Article Language.', data_type=DataType.TEXT,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='labels', description='Article Matched Keyword Expression ID.', data_type=DataType.TEXT_ARRAY,
                    index_searchable=True, index_filterable=True, skip_vectorization=True
                ),
                Property(
                    name='text', description='The article content.', data_type=DataType.TEXT,
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
    ./mulabel db init -c mulabel
    ./mulabel db init -c mulabel -l sl
    ./mulabel db init -c mulabel -l sl,sr
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
    ./mulabel db drop -c mulabel
    ./mulabel db drop -c mulabel -l sl,sr
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
    ./mulabel db pump -c mulabel
    ./mulabel db pump -c mulabel -l sl,sr

    ./mulabel db drop -c mulabel -l sl --public
    ./mulabel db init -c mulabel -l sl --public
    ./mulabel db pump -c mulabel -l sl --public
    ./mulabel db pump -c whole_mulabel -l sl --public

    ./mulabel db pump -c mulabel -l sl --public --seed_only
    """
    lpr = True
    result = arg.collection.removeprefix('whole_')
    if result != arg.collection:
        lpr = False
        arg.collection = result

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

    client = weaviate.connect_to_local(port=__WEAVIATE_PORT)
    coll_name = arg.collection if lpr else 'whole_' + arg.collection
    try:
        if not client.collections.exists(coll_name):
            logger.warning('Collection [%s] does not exist.', coll_name)
            return 1
        coll = client.collections.get(coll_name)
        stored = set()
        for lrp_dict in lrp_dicts:
            record_uuid = generate_uuid5(lrp_dict)
            if record_uuid in stored:
                continue

            logger.info('Processing label [%s]', lrp_dict)
            vectors = {}
            for model_name, model in models.items():
                if lpr:
                    vectors['m_' + model_name] = model(lrp_dict['passage'])
                else:
                    vectors['m_' + model_name] = model(lrp_dict['text'])

            coll.data.insert(
                uuid=record_uuid,
                properties=lrp_dict,
                vector=vectors
            )
            stored.add(record_uuid)
    finally:
        client.close()

    return 0


def db_ps_calib_bge_m3(arg) -> int:
    """
    ./mulabel db calib_bge_m -c mulabel
    ./mulabel db calib_bge_m -c mulabel -l sl,sr

    ./mulabel db calib_bge_m -c mulabel -l sl --public
    ./mulabel db calib_bge_m -c mulabel -l sl --public --seed_only
    """
    compute_arg_collection_name(arg)
    tokenizers = {}
    for lang in arg.lang:
        tokenizers[lang] = get_segmenter(lang, arg.data_in_dir)

    bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    lrp_file_name = os.path.join(arg.data_in_dir, f'{arg.collection}_filtered_lrp_dev.csv')
    lrp_df = pd.read_csv(lrp_file_name)
    if ptypes.is_string_dtype(lrp_df['kwe_id']):
        lrp_df['kwe_id'] = lrp_df['kwe_id'].apply(ast.literal_eval)
    if ptypes.is_string_dtype(lrp_df['kwe']):
        lrp_df['kwe'] = lrp_df['kwe'].apply(ast.literal_eval)
    lrp_df = lrp_df[lrp_df['passage_cat'] == arg.passage_size]
    lrp_dicts = lrp_df.to_dict(orient='records')

    calibration = {}
    client = weaviate.connect_to_local(port=__WEAVIATE_PORT)
    try:
        if not client.collections.exists(arg.collection):
            logger.warning('Collection [%s] does not exist.', arg.collection)
            return 1
        coll = client.collections.get(arg.collection)
        for idx, lrp in enumerate(lrp_dicts):
            # logger.info('Processing lrp [%s]', lrp)

            query_vector = bge_m3_model.encode([lrp['passage']])['dense_vecs'].tolist()[0]
            response = coll.query.near_vector(
                near_vector=query_vector,  # your query vector goes here
                limit=10,
                filters=(
                    Filter.by_property('passage_cat').equal(lrp['passage_cat']) &
                    Filter.by_property('label_id').equal(lrp['label_id'])
                ),
                return_metadata=MetadataQuery(distance=True)
            )
            # print(f'Matching: [{lrp["passage"]}')
            if not lrp['label_id'] in calibration:
                calibration[lrp['label_id']] = {
                    'id': lrp['label_id'], 'label': '', 'max': 0, 'top_values': []
                }
            if idx % 1000 == 0:
                logger.info('Processing lrp [%s/%s]', idx, len(lrp_dicts))
            for o in response.objects:
                if o.metadata.distance < 0.001:  # skip duplicate
                    continue
                calibration[lrp['label_id']]['top_values'].append(o.metadata.distance)
                if o.metadata.distance > calibration[lrp['label_id']]['max']:
                    calibration[lrp['label_id']]['max'] = o.metadata.distance
                break
                #print(f'[{o.properties["label_id"]}|cat:{o.properties["passage_cat"]}|a:{o.properties["a_id"]}]: '
                #      f'{o.metadata.distance} [{o.properties["passage"]}]')
    finally:
        client.close()
    calibration_df = pd.DataFrame(calibration.values())
    calibration_df.to_csv(f'{arg.collection}_ps{arg.passage_size}_calibration.csv', index=False)
    return 0


def group_and_concat(input_list, group_size, separator=''):
    return [separator.join(map(str, input_list[i:i+group_size]))
            for i in range(0, len(input_list), group_size)]


def db_test_bge_m3(arg) -> int:
    """
    ./mulabel db test_bge_m3 -c mulabel
    ./mulabel db test_bge_m3 -c mulabel -l sl,sr

    ./mulabel db test_bge_m3 -c mulabel -l sl --public
    ./mulabel db test_bge_m3 -c mulabel -l sl --public --seed_only
    """
    lpr = True
    result = arg.collection.removeprefix('whole_')
    if result != arg.collection:
        lpr = False
        arg.collection = result

    compute_arg_collection_name(arg)
    tokenizers = {}
    for lang in arg.lang:
        tokenizers[lang] = get_segmenter(lang, arg.data_in_dir)

    bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    test_file_name = os.path.join(arg.data_in_dir, f'{arg.collection}_filtered_article_test.csv')
    test_df = pd.read_csv(test_file_name)
    if ptypes.is_string_dtype(test_df['labels']):
        test_df['labels'] = test_df['labels'].apply(ast.literal_eval)
    test_dicts = test_df.to_dict(orient='records')

    labels_file_name = os.path.join(arg.data_in_dir, f'labels.csv')
    with open(labels_file_name, 'r') as l_file:
        all_labels = [line.strip() for line in l_file]

    labeler = MultilabelLabeler(all_labels)
    labeler.fit()

    client = weaviate.connect_to_local(port=__WEAVIATE_PORT)
    coll_name = arg.collection if lpr else 'whole_' + arg.collection
    y_true = []
    y_pred = []
    try:
        if not client.collections.exists(coll_name):
            logger.warning('Collection [%s] does not exist.', coll_name)
            return 1
        coll = client.collections.get(coll_name)
        for idx, article in enumerate(test_dicts):
            logger.info('Processing article [%s]', article)

            doc = tokenizers[article['lang']](article['text'])
            sentences = [sentence.text for sentence in doc.sentences]
            chunks = group_and_concat(sentences, arg.passage_size, separator=' ')
            chunks = [article['text']]
            query_vectors = bge_m3_model.encode(chunks)['dense_vecs'].tolist()
            for query_vector, chunk in zip(query_vectors, chunks):
                if lpr:
                    response = coll.query.near_vector(
                        near_vector=query_vector,  # dense representation
                        limit=50,
                        distance=0.50,
                        filters=Filter.by_property("passage_cat").equal(arg.passage_size),
                        return_metadata=MetadataQuery(distance=True)
                    )
                    print('===================================================================')
                    print(f'Matching: [{chunk}]')
                    labels = {}
                    for o in response.objects:
                        if not o.properties["label_id"] in labels:
                            labels[o.properties["label_id"]] = o.metadata.distance
                        elif labels[o.properties["label_id"]] > o.metadata.distance:
                            labels[o.properties["label_id"]] = o.metadata.distance
                        #print(f'[{o.properties["label_id"]}|cat:{o.properties["passage_cat"]}|a:{o.properties["a_id"]}]: '
                        #      f'{o.metadata.distance} [{o.properties["passage"]}]')
                    print(f'{labels}\n')
                else:
                    response = coll.query.near_vector(
                        near_vector=query_vector,  # dense representation
                        limit=10,
                        distance=0.40,
                        return_metadata=MetadataQuery(distance=True)
                    )
                    print('===================================================================')
                    y_true.append(labeler.vectorize([article['labels']])[0])
                    print(f'[{article["labels"]}]')
                    if len(response.objects) == 0:
                        y_pred.append(labeler.vectorize([[]])[0])
                    else:
                        for o in response.objects:
                            print(f'[{o.properties["labels"]}|a:{o.properties["a_id"]}]: '
                                  f'{o.metadata.distance} [{o.properties["text"][:-1]}]')
                            y_pred.append(labeler.vectorize([o.properties["labels"]])[0])
                            break
            if idx > 10:
                break

    finally:
        client.close()

    average_type = 'micro'
    p = precision_score(y_true, y_pred, average=average_type)
    r = recall_score(y_true, y_pred, average=average_type)
    f1 = f1_score(y_true, y_pred, average=average_type)
    print(f'Precision:{p}\nRecall:{r}\nF1:{f1}')
    return 0
