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
from tqdm import tqdm

from ...core.args import CommonArguments
from ...core.labels import MultilabelLabeler, Labeler
from ...core.metrics import Metrics
from ...core.wandb import initialize_run
from ..tokenizer import get_segmenter
from ..utils import (__supported_languages, __supported_passage_sizes, compute_arg_collection_name,
                     load_add_corpus_part, parse_arg_passage_sizes)
from .utils import load_data, State, SimilarParams, find_similar, LabelStats, LabelDecider

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
        '--passage_sizes', help=f'When calibrating use passage_sizes '
                                f'You can use a comma separated list of {__supported_passage_sizes}',
        type=str,
    )
    parser.add_argument(
        '--run_id', type=int, help=f'Run id for marking consecutive runs.', default=0
    )
    parser.add_argument(
        '--calib_max', type=int, help=f'Max number of labels to calibrate on.', default=-1
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

    duplicates = []
    dup_checked = set()
    client = Elasticsearch(CLIENT_URL)
    try:
        def on_similar(s: State) -> bool:
            if s.score > 0.99:
                # logger.info(
                #    'Article [%s] [%s][%s] is very similar to [%s][%s]',
                #    state['count'], q_a_uuid, state['doc']['date'], v_a_uuid, data_item['date']
                # )
                # logger.info('Similar texts: \n\n[%s]\n\n[%s]', state['doc']['text'], data_item['text'])
                dup_checked.add(s.hit['a_uuid'])

                del s.hit['m_bge_m3']
                s.hit['similar_uuid'] = s.item['a_uuid']
                s.hit['similar_id'] = s.item['a_id']
                duplicates.append(s.hit)
            return True

        def on_item(data_item: Dict[str, Any]) -> bool:
            if data_item['a_uuid'] in dup_checked:
                return True
            s = State(data_item, 'text')
            embedding = data_item['m_bge_m3']
            params = SimilarParams(args.collection, data_item['a_uuid'], embedding)
            find_similar(client, params, s, on_similar)
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

    y_true = []
    y_pred = []
    client = Elasticsearch(CLIENT_URL)
    try:
        if not client.indices.exists(index=train_coll_name):
            logger.warning('Collection [%s] does not exist.', train_coll_name)
            return 1

        def on_similar(curr_state: State) -> bool:
            y_pred.append(labeler.vectorize([curr_state.hit['label']])[0])
            y_true.append(labeler.vectorize([curr_state.item["label"]])[0])
            return False

        for data_item in tqdm(data_as_dicts, desc='Processing zero shot complete eval.'):
            state = State(data_item, 'text')
            embedding = model(state.text)[0].tolist()
            params = SimilarParams(train_coll_name, data_item['a_uuid'], embedding)
            num_ret = find_similar(client, params, state, on_similar)
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
    parse_arg_passage_sizes(args)
    output_name = f'{args.collection}_calib'
    run = _init_task(args, output_name, 'BAAI/bge-m3', 'bge_m3')
    labeler = _init_labeler(args)
    models = _init_ebd_models(args)
    model = models[args.ptm_alias]

    train_coll = args.collection + '_train'
    dev_coll = args.collection + '_dev'
    data_as_dicts = _load_data(args, dev_coll)
    data_as_dicts.extend(_load_data(args, train_coll))

    def on_similar(state: State) -> bool:
        if label not in state.hit['label']:
            y_prob.append(state.score)
            y_true.append(0)
            return True
        y_prob.append(state.score)
        y_true.append(1)
        return True

    client = Elasticsearch(CLIENT_URL)
    try:
        if not client.indices.exists(index=train_coll):
            logger.warning('Collection [%s] does not exist.', train_coll)
            return 1
        df = pd.DataFrame(data=data_as_dicts)
        for passage_size in args.passage_sizes:
            label_thresholds = {}
            for i, label in tqdm(
                    enumerate(labeler.labels), f'Processing labels for passage category {passage_size}:'
            ):
                filtered = df[(df['passage_cat'].isin([passage_size, 0])) & (df['label'].apply(lambda x: label in x))]
                y_true = []
                y_prob = []
                passage_data_as_dicts = filtered.to_dict(orient='records')
                states = []
                for data_i, data_item in enumerate(passage_data_as_dicts):
                    state = State(data_item, 'text')
                    states.append(state)
                    embedding = model(state.text)[0].tolist()
                    passage_cat = data_item['passage_cat']

                    if passage_cat == 0:
                        params = SimilarParams(train_coll, data_item['a_uuid'], embedding, passage_cat=0)
                        find_similar(client, params, state, on_similar)
                    else:
                        params = SimilarParams(
                            train_coll, data_item['a_uuid'], embedding, passage_targets=[passage_cat]
                        )
                        find_similar(client, params, state, on_similar)
                    if state.total == 0:
                        continue

                if label not in label_thresholds:
                    label_thresholds[label] = LabelStats(label)
                if len(y_prob) > 0:
                    label_thresholds[label].compute(y_true, y_prob)

                if i % 100 == 0:
                    logger.info(f'At label {i}/{label}')
                if 0 < args.calib_max <= i:
                    break

            label_threshold_list = [v.__dict__ for v in label_thresholds.values()]
            calib_cat_file_name = os.path.join(args.data_in_dir, f'{output_name}_ps{passage_size}.csv')
            pd.DataFrame(data=label_threshold_list).to_csv(calib_cat_file_name, index=False, encoding='utf-8')
    finally:
        client.close()

    return 0


def es_test_lrp_bge_m3(args) -> int:
    """
    In development - specifically for LRP
    ./mulabel es test_lrp_bge_m3 -c lrp_mulabel -l sl --passage_sizes 1
    ./mulabel es test_lrp_bge_m3 -c lrp_mulabel -l sl --public --passage_sizes 1
    ./mulabel es test_lrp_bge_m3 -c lrp_mulabel -l sl --public --seed_only --passage_sizes 1
    """
    compute_arg_collection_name(args)
    parse_arg_passage_sizes(args)
    if len(args.passage_sizes) > 1:
        raise ValueError('There can be only one passage size for testing ... sorry!')

    passage_size = args.passage_sizes[0]
    calib_cat_file_name = os.path.join(args.data_in_dir, f'{args.collection}_calib_ps{passage_size}.csv')
    if not os.path.exists(calib_cat_file_name):
        raise ValueError(f'You need to calibrate labels first to produce [{calib_cat_file_name}] file! ... sorry!')

    calib_df = pd.read_csv(calib_cat_file_name, encoding='utf-8')

    def dict_to_stats_obj(l: str, d: Dict[str, Any]) -> LabelStats:
        stat = LabelStats(l)
        stat.__dict__.update(d)
        return stat

    calib_dict: Dict[str, LabelStats] = {row['label']: dict_to_stats_obj(
        row['label'], row.drop('label').to_dict()
    ) for _, row in calib_df.iterrows()}

    output_name = f'zshot_{passage_size}_' + args.collection
    run = _init_task(args, output_name, 'BAAI/bge-m3', 'bge_m3')
    labeler = _init_labeler(args)
    metrics = Metrics(output_name, labeler.get_type_code())

    segmenters = _init_segmenters(args)
    models = _init_ebd_models(args)
    model = models[args.ptm_alias]

    train_coll = args.collection + '_train'
    test_coll = args.collection + '_test'
    data_as_dicts = _load_data(args, test_coll)

    y_true = []
    y_pred = []
    client = Elasticsearch(CLIENT_URL)
    try:
        if not client.indices.exists(index=train_coll):
            logger.warning('Collection [%s] does not exist.', train_coll)
            return 1

        for data_item in tqdm(data_as_dicts, desc='Processing zero shot LRP eval.'):
            passage_cat = data_item['passage_cat']
            if passage_cat != passage_size and passage_cat != 0:
                continue
            decider = LabelDecider(data_item['label'], calib_dict)
            if decider.skip():  # no labels in calibration
                continue

            def on_similar(s: State) -> bool:
                pred_labels = s.hit['label'].copy()
                compare_score = 0.80
                s.pop()  # remove similar
                for pred_label in pred_labels:
                    if pred_label not in calib_dict:
                        return True

                    # predicted was calibrated
                    pred_stats: LabelStats = calib_dict[pred_label]
                    if pred_stats.num > 0 and pred_stats.pos > 0 and pred_stats.neg > 0:
                        compare_score = pred_stats.opt

                    if s.score >= compare_score and not pred_label in s.data['label']:
                        s.data['label'].append(pred_label)

                return True

            state = State(data_item, 'text')
            state.data['label'] = []
            lang = data_item['lang']

            if lang not in segmenters:
                logger.warning('Language [%s] does not exist.', lang)
                continue

            for sentence in segmenters[lang](data_item['text']).sentences:
                state.text = sentence.text
                vec = model(state.text)[0].tolist()

                if passage_cat == 0:
                    params = SimilarParams(train_coll, data_item['a_uuid'], vec, passage_cat=0)
                    find_similar(client, params, state, on_similar)
                else:
                    params = SimilarParams(
                        train_coll, data_item['a_uuid'], vec, passage_targets=[data_item['passage_cat']]
                    )
                    find_similar(client, params, state, on_similar)

            y_pred.append(labeler.vectorize([state.data['label']])[0])
            y_true.append(labeler.vectorize([decider.calibrated.keys()])[0])

    finally:
        client.close()

    m = metrics(np.array(y_true, dtype=float), np.array(y_pred, dtype=float), 'test/')
    metrics.dump(args.data_result_dir, None, run)
    if run is not None:
        run.log(m)

    return 0
