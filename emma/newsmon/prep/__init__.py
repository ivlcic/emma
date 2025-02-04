import os
import logging
import time
import pandas as pd

from typing import Dict, Any, List
from argparse import ArgumentParser
from tqdm import tqdm

from ...core.args import CommonArguments
from ...core.models import retrieve_model_name_map

from ..tokenizer import get_segmenter
from ..utils import compute_arg_collection_name, init_labeler, load_data, chunk_data
from ..const import __supported_languages, __label_split_names, __supported_passage_sizes


logger = logging.getLogger('newsmon.prep')


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
        '--lrp_size', type=int, help=f'LRP size.', default=1
    )
    parser.add_argument(
        '--test_l_class', type=str, help=f'Test specified label class.',
        choices=['all'].extend(__label_split_names), default='all'
    )
    parser.add_argument(
        '--ptm_models',
        help=f'Use only ptm_models (filter everything else out). '
             f'You can use a comma separated list of {retrieve_model_name_map.keys()}',
        type=str,
        default=next(iter(retrieve_model_name_map))
    )



def _init_segmenters(args) -> Any:
    tokenizers = {}
    for lang in args.lang:
        tokenizers[lang] = get_segmenter(lang, args.tmp_dir)
    return tokenizers


def __find_label_in_text(kwes: List[str], doc) -> List[str]:
    passages = []
    for sentence in doc.sentences:
        for kwe in kwes:
            if kwe in sentence.text:
                passages.append(sentence.text)
    return passages


def prep_lrp_extract(args) -> int:
    """
    ./mulabel fa lrp_extract -c mulabel -l sl --public
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir
    compute_arg_collection_name(args)
    labeler = init_labeler(args)
    all_labels = set(labeler.labels)
    for t in ['train', 'dev', 'test']:
        _, df = load_data(args, f'lrp_{args.collection}_{t}')  # we load the data
        df = df[(df['passage_cat'] == 0) | (df['passage_cat'] == args.lrp_size)]
        filtered_dicts = df.to_dict(orient='records')
        filtered_data = []
        for f in filtered_dicts:  # remove samples with labels not in all_labels
            f['label'] = [label for label in f['label'] if label in all_labels]
            if len(f['label']) > 0:
                filtered_data.append(f)

        df = pd.DataFrame(filtered_data)
        df.to_csv(
            os.path.join(args.data_in_dir, f'lrp-{args.lrp_size}_{args.collection}_{t}.csv'),
            index=False, encoding='utf-8'
        )

    return 0


def prep_init_pseudo_labels(args) -> int:
    """
    ./mulabel fa init_pseudo_labels -c mulabel -l sl --public
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    segmenters = _init_segmenters(args)

    # now we construct pseudo label descriptions
    label_dict: Dict[str, Dict['str', Any]] = {}
    data_as_dicts, _ = load_data(args, 'lrp_' + args.collection + '_train')  # we load the train data
    for item in data_as_dicts:
        if item['passage_cat'] != 0 and item['passage_cat'] != 1:
            continue
        for k, label in enumerate(item['label']):
            if label not in label_dict:
                label_dict[label] = {
                    'label': label,
                    'label_info': item['label_info'][k],
                    'texts': [], 'passages': []
                }
            if item['passage_cat'] == 1:
                label_dict[label]['passages'].append(item['text'])
            else:
                passages = []
                if item['lang'] in segmenters:
                    search = []
                    if 'kwe' in item['label_info'][k]:
                        search.extend([k['value'] for k in item['label_info'][k]['kwe'] if 'value' in k])
                    if 'name' in item['label_info'][k]:
                        search.append(item['label_info'][k]['name'])

                    if search:
                        if 'doc' not in item:
                            segmenter = segmenters[item['lang']]
                            item['doc'] = segmenter(item['text'])
                        passages = __find_label_in_text(search, item['doc'])
                if passages:
                    label_dict[label]['passages'].extend(passages)
                else:
                    label_dict[label]['texts'].append(item['text'])

    labeler = init_labeler(args)
    labels_df_data = []
    for id, label in labeler.ids_to_labels().items():
        if label not in label_dict:
            logger.warning(f'Missing any text for label {label}')
            continue
        label_data = label_dict[label]
        label_data['id'] = id
        labels_df_data.append(label_data)

    labels_df = pd.DataFrame(labels_df_data)
    labels_df.to_csv(os.path.join(
        args.data_in_dir, f'{args.collection}_labels_descr.csv'), index=False, encoding='utf-8')

    return 0


def prep_export_label_space(args) -> int:
    """
    ./mulabel fa export_label_space -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    labeler = init_labeler(args)

    labels_map_filename = os.path.join(args.data_out_dir, args.collection + '_map_labels.csv')
    labels_map_df = pd.read_csv(str(labels_map_filename), encoding='utf-8')
    labels_maps = labels_map_df.to_dict(orient='records')
    labels_map = {item['id']: item for item in labels_maps}


    train_coll_name = args.collection + '_train'
    dev_coll_name = args.collection + '_dev'
    data_as_dicts, _ = load_data(args, train_coll_name)
    data = []
    t0 = time.time()
    for chunk in tqdm(chunk_data(data_as_dicts, chunk_size=384), desc='Processing zero shot eval.'):
        doc_ids = [item['a_id'] for item in chunk]
        yl_true = labeler.vectorize([item['label'] for item in chunk]).tolist()
        for i in range(len(yl_true)):
            yl_true[i].insert(0, doc_ids[i])
        data.extend(yl_true)

    logger.info(f'Measured performance in {(time.time() - t0):8.2f} seconds')


    headers = ['doc_id']
    for i in range(labeler.num_labels):
        label_id = labeler.encoder.classes_[i]
        label_name = labels_map[label_id]['name']
        headers.append(label_name)

    df = pd.DataFrame(data, columns=headers)

    file_path = 'exported_label_space.tsv'
    df.to_csv(os.path.join(args.data_result_dir, file_path), sep='\t', index=False, encoding='utf-8')

    logger.info(f'Computation done in {(time.time() - t0):8.2f} seconds')
    return 0
