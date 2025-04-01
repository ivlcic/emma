import os
import ast
import logging
import time
import random
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from typing import Dict, Any, List
from argparse import ArgumentParser
from tqdm import tqdm

from ..embd_model import EmbeddingModelWrapperFactory
from ...core.args import CommonArguments
from ...core.models import retrieve_model_name_map

from ..tokenizer import get_segmenter
from ..utils import compute_arg_collection_name, init_labeler, load_data, chunk_data, get_index_path
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


# noinspection DuplicatedCode
def prep_lrp_extract(args) -> int:
    """
    ./newsmon fa lrp_extract -c mulabel -l sl --public
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


# noinspection DuplicatedCode
def prep_init_pseudo_labels(args) -> int:
    """
    ./newsmon fa init_pseudo_labels -c mulabel -l sl --public
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


# noinspection DuplicatedCode
def _select_random_sentences(passages, max_chars):
    selected_sentences = []
    total_chars = 0

    # Shuffle the list to ensure randomness
    random.shuffle(passages)

    for sentence in passages:
        # Check if adding the current sentence exceeds the max character limit
        if total_chars + len(sentence) <= max_chars:
            selected_sentences.append(sentence)
            total_chars += len(sentence)
        else:
            break

    return ' '.join(selected_sentences)


# noinspection DuplicatedCode
def prep_init_rae_v(args) -> int:
    """
    ./newsmon prep init_rae_v -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler = init_labeler(args)

    # read label descriptions / passages
    label_descr_file_path = os.path.join(args.data_in_dir, f'{args.collection}_labels_descr.csv')
    if not os.path.exists(label_descr_file_path):
        logger.warning(f'Missing label description file [{label_descr_file_path}]. '
                       f'Run [./newsmon prep init_pseudo_labels -c mulabel -l sl --public] or similar first!')
        return 1
    label_descr_df = pd.read_csv(label_descr_file_path)
    label_descr_df['passages'] = label_descr_df['passages'].apply(ast.literal_eval)
    label_descr_df['texts'] = label_descr_df['texts'].apply(ast.literal_eval)
    label_descr_df['label_info'] = label_descr_df['label_info'].apply(ast.literal_eval)

    random.seed(2611)
    num_labels = labeler.num_labels
    descr_size_chars = 2000
    labels: Dict[str: Dict[str, Any]] = {}
    for model_name in models:
        labels[model_name] = {}
        labels[model_name]['samples'] = num_labels
        labels[model_name]['v_train'] = []
        labels[model_name]['y_id'] = []

    texts = ['_'] * num_labels
    label_id_map = labeler.labels_to_ids()
    for label in label_descr_df.to_dict('records'):
        if label['passages'] and label['texts']:
            text = _select_random_sentences(label['passages'], descr_size_chars)
        elif label['passages']:
            text = _select_random_sentences(label['passages'], descr_size_chars)
        else:
            random.shuffle(label['texts'])
            text = label['texts'][0]
        l_id = label_id_map[label['label']]
        texts[l_id] = text

    for chunk in chunk_data(texts, chunk_size=384):
        for model_name, model in models.items():
            ret = model.embed(chunk)
            labels[model_name]['v_train'].append(ret)

    index_path = get_index_path(args)
    for model_name in models:
        data_dict = labels[model_name]
        data_dict['v_train'] = np.vstack(labels[model_name]['v_train'])
        data_dict['y_id'] = np.identity(num_labels)
        # noinspection PyTypeChecker
        np.savez_compressed(index_path + '.' + model_name + '_v.npz', **data_dict)

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


def prep_analyze(arg) -> int:
    """
    Analyzes map files
    ./mulabel prep analyze --postfix 2023_01,2023_02
    """
    logger.debug("Analyzes map newsmon article files.")
    os.environ['HF_HOME'] = arg.data_out_dir  # local tmp dir

    if ',' in arg.postfix:
        arg.postfix = arg.postfix.split(',')
    else:
        arg.postfix = [arg.postfix]

    l_col = 'tags'
    dfs = []
    for postfix in arg.postfix:
        article_map_file_name = os.path.join(arg.data_in_dir, 'map', f'map_articles_{postfix}.csv')
        logger.info(f'Reading file %s', article_map_file_name)
        tmp_df = pd.read_csv(article_map_file_name, encoding='utf-8')
        if ptypes.is_string_dtype(tmp_df[l_col]):
            tmp_df[l_col] = tmp_df[l_col].apply(ast.literal_eval)
        elif ptypes.is_integer_dtype(tmp_df[l_col]):
            tmp_df[l_col] = tmp_df[l_col].apply(lambda x: [x])
        dfs.append(tmp_df)


    df = pd.concat(dfs, ignore_index=True)
    print(df.head())
    all_labels = set()
    lang_tag_counts = {}
    for lang in df['lang'].unique():
        lang_tags = [tag for tags_list in df[df['lang'] == lang]['tags'] for tag in tags_list]
        all_labels.update(lang_tags)
        lang_tag_counts[lang] = len(set(lang_tags))
    total = 0
    for k, v in lang_tag_counts.items():
        total += v
    lang_tag_counts['all'] = len(all_labels)
    lang_tag_counts['total'] = total
    print(lang_tag_counts)

    sample_counts = df['lang'].value_counts().to_dict()
    total = 0
    for k, v in sample_counts.items():
        total += v
    sample_counts['total'] = total
    print(sample_counts)

    all_tokens = df['sp_tokens']

    # Define bins (0 to 8192 with increments of 512)
    bins = list(range(1, 8194, 512))  # Includes the upper bound (8192)

    # Calculate histogram using numpy
    hist, bin_edges = np.histogram(all_tokens, bins=bins)

    # Create a DataFrame for better visualization of bin ranges and frequencies
    histogram_df = pd.DataFrame({
        'Range': [f"{bin_edges[i]}-{bin_edges[i + 1] - 1}" for i in range(len(bin_edges) - 1)],
        'Frequency': hist
    })

    # Calculate the total number of samples
    total_samples = histogram_df['Frequency'].sum()
    print(total_samples)
    # Calculate the percentage for each bin
    histogram_df['Percentage'] = (histogram_df['Frequency'] / total_samples) * 100

    print(histogram_df)

    max_limit = 4097
    bin_step = 512
    bins = list(range(1, max_limit + bin_step, bin_step)) + [np.inf]

    # Extract token counts
    token_counts = df['sp_tokens']
    # Calculate histogram
    hist, bin_edges = np.histogram(token_counts, bins=bins)

    # Create readable bin labels
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if upper == np.inf:
            bin_labels.append(f"{lower}-inf")
        else:
            bin_labels.append(f"{lower}-{upper - 1}")

    # Create DataFrame with percentages
    total_samples = len(token_counts)
    histogram_df = pd.DataFrame({
        'Token Range': bin_labels,
        'Count': hist,
        'Percentage': np.round(hist / total_samples * 100, 2)
    })
    print(histogram_df)

    return 0
