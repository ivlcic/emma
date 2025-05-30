import glob
import os
import ast
import logging
import time
import random
from collections import Counter

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
from ..utils import compute_arg_collection_name, init_labeler, load_data, chunk_data, get_index_path, load_labels
from ..const import __supported_languages, __label_split_names, __supported_passage_sizes, __label_splits

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
    labeler, _ = init_labeler(args)
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

    labeler, _ = init_labeler(args)
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
    labeler, _ = init_labeler(args)

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
    labeler, _ = init_labeler(args)

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


def prep_label_project(args):
    """
    UMAP label space projection
    ./newsmon prep label_project -c newsmon -l sl --public
    ./newsmon prep label_project -c newsmon -l sl --public --test_l_class Rare
    ./newsmon prep label_project -c newsmon -l sl --public --test_l_class Frequent

    ./newsmon prep label_project -c eurlex
    ./newsmon prep label_project -c eurlex --test_l_class Rare
    ./newsmon prep label_project -c eurlex --test_l_class Frequent
    """
    compute_arg_collection_name(args)
    labeler, labels_df = init_labeler(args)
    train_data_as_dicts, train_df = load_data(args, args.collection + '_train')  # we load the train data
    dev_data_as_dicts, dev_df = load_data(args, args.collection + '_dev')  # we load the validation data
    test_data_as_dicts, test_df = load_data(args, args.collection + '_test')  # we load the test data

    data_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    suffix = ''
    c = args.collection_conf
    if 'newsmon' in args.collection:
        c = 'NewsMon'
    if 'eurlex' in args.collection:
        c = 'EurLex57K'
    title_append = f' - {c}'
    target_labels = None
    if args.test_l_class != 'all':
        title_append += f', {args.test_l_class} labels'
        suffix = '_' + args.test_l_class
        label_classes = load_labels(args.data_in_dir, args.collection, __label_splits, __label_split_names)
        target_labels = label_classes[args.test_l_class]

    sample_tag_counts = []
    for instance_labels in data_df['label']:
        instance_labels.sort()
        if target_labels:
            instance_labels = [item for item in instance_labels if item in target_labels]
        if not instance_labels:
            continue
        sample_tag_counts.append(len(instance_labels))

    std_dev_labels = np.std(sample_tag_counts, axis=0)
    mean_labels = np.mean(sample_tag_counts, axis=0)
    total = len(sample_tag_counts)
    logger.info(f'Mean {mean_labels} Standard deviation {std_dev_labels}, total samples {total}')

    label_lists = []
    for split in [train_data_as_dicts, dev_data_as_dicts, test_data_as_dicts]:
        for data_sample in split:
            label_lists.append(data_sample['label'])

    label_space = np.array(labeler.vectorize(label_lists))
    logger.info(f'Label space {label_space.shape}')

    label_density = np.sum(label_space, axis=1)  / label_space.shape[1]
    logger.info(f'Density space {label_density.shape}')
    logger.info(f'Density space {label_density}')
    max_dens = np.max(label_density)
    min_dens = np.min(label_density)
    bins = np.linspace(min_dens, max_dens, 11)
    logger.info(f'Bins {bins.shape}{bins}')
    digitized_samples = np.digitize(label_density, bins, right=False) - 1
    logger.info(f'Digitized space {digitized_samples.shape}')
    logger.info(f'Digitized space {digitized_samples}')

    #import cupy as cp
    #from cuml.manifold import UMAP
    import umap
    import seaborn as sns

    # Initialize the UMAP model
    umap_model = umap.UMAP()

    t0 = time.time()
    embedding = umap_model.fit_transform(label_space)
    logger.info(f'Computation done in {(time.time() - t0):8.2f} seconds')
    logger.info(f'Embedding space {embedding.shape}')

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0],
                    y=embedding[:, 1],
                    hue=digitized_samples,
                    size=1,
                    #hue_order=np.arange(0, 10, 1, dtype=int),
                    palette="coolwarm",
                    edgecolor=None,
                    alpha=0.7)
    plt.gca().set_aspect('equal', 'datalim')
    #plt.title(f'UMAP Visualization{title_append}', fontsize=18)
    plt.legend(title="Label Density", ncol=4)
    plt.tight_layout()
    output_filename = os.path.join(args.data_result_dir, f'label_tsne_proj_{args.collection}{suffix}.png')
    plt.savefig(output_filename, dpi=300)  # , transparent=True)
    plt.close()


def prep_label_project2(args):
    """
    UMAP label space projection
    ./newsmon prep label_project2 -c newsmon -l sl --public
    ./newsmon prep label_project2 -c newsmon -l sl --public --test_l_class Rare
    ./newsmon prep label_project2 -c newsmon -l sl --public --test_l_class Frequent

    ./newsmon prep label_project2 -c eurlex
    ./newsmon prep label_project2 -c eurlex --test_l_class Rare
    ./newsmon prep label_project2 -c eurlex --test_l_class Frequent
    """
    compute_arg_collection_name(args)
    labeler, labels_df = init_labeler(args)
    train_data_as_dicts, train_df = load_data(args, args.collection + '_train')  # we load the train data
    dev_data_as_dicts, dev_df = load_data(args, args.collection + '_dev')  # we load the validation data
    test_data_as_dicts, test_df = load_data(args, args.collection + '_test')  # we load the test data

    data_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    suffix = ''
    c = args.collection_conf
    if 'newsmon' in args.collection:
        c = 'NewsMon'
    if 'eurlex' in args.collection:
        c = 'EurLex57K'
    title_append = f' - {c}'
    target_labels = None
    if args.test_l_class != 'all':
        title_append += f', {args.test_l_class} labels'
        suffix = '_' + args.test_l_class
        label_classes = load_labels(args.data_in_dir, args.collection, __label_splits, __label_split_names)
        target_labels = label_classes[args.test_l_class]

    sample_tag_counts = []
    for instance_labels in data_df['label']:
        instance_labels.sort()
        if target_labels:
            instance_labels = [item for item in instance_labels if item in target_labels]
        if not instance_labels:
            continue
        sample_tag_counts.append(len(instance_labels))

    std_dev_labels = np.std(sample_tag_counts, axis=0)
    mean_labels = np.mean(sample_tag_counts, axis=0)
    total = len(sample_tag_counts)
    logger.info(f'Mean {mean_labels} Standard deviation {std_dev_labels}, total samples {total}')

    label_lists = []
    for split in [train_data_as_dicts, dev_data_as_dicts, test_data_as_dicts]:
        for data_sample in split:
            label_lists.append(data_sample['label'])

    label_space = np.array(labeler.vectorize(label_lists))
    logger.info(f'Label space {label_space.shape}')

    label_density = np.sum(label_space, axis=1)  / label_space.shape[1]
    logger.info(f'Density space {label_density.shape}')
    logger.info(f'Density space {label_density}')
    max_dens = np.max(label_density)
    min_dens = np.min(label_density)
    bins = np.linspace(min_dens, max_dens, 11)
    logger.info(f'Bins {bins}')
    digitized_samples = np.digitize(label_density, bins, right=False) - 1
    logger.info(f'Digitized space {digitized_samples.shape}')
    logger.info(f'Digitized space {digitized_samples}')

    #import cupy as cp
    #from cuml.manifold import UMAP
    import umap
    import seaborn as sns

    # Initialize the UMAP model
    umap_model = umap.UMAP()

    t0 = time.time()
    embedding = umap_model.fit_transform(label_space)
    logger.info(f'Computation done in {(time.time() - t0):8.2f} seconds')
    logger.info(f'Embedding space {embedding.shape}')

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    for bin_idx, _ in enumerate(bins):
        # Mask for current density bin
        mask = digitized_samples == bin_idx
        #if (bin_idx) % 2 == 1:
        #    continue
        if not np.any(mask):  # Skip empty bins
            continue

        # Get points for this density bin
        x_vals = embedding[mask, 0]
        y_vals = embedding[mask, 1]
        masked_hue = digitized_samples[mask]

        plt.scatter(
            x=x_vals,
            y=y_vals,
            c=[bin_idx] * len(x_vals),  # Array matching x/y length
            #hue=masked_hue,
            cmap="coolwarm",
            vmin=0,
            vmax=len(bins) - 1,
            edgecolor="none",
            alpha=0.7,
            s=10,
            zorder=bin_idx,
            label=bin_idx
        )

    plt.gca().set_aspect('equal', 'datalim')
    #plt.title(f'UMAP Visualization{title_append}', fontsize=18)

    plt.legend(title="Label Density", loc="upper right", ncol=1, markerscale=3)
    plt.tight_layout()
    output_filename = os.path.join(args.data_result_dir, f'label_tsne_proj2_{args.collection}{suffix}.png')
    plt.savefig(output_filename, dpi=300)  # , transparent=True)
    plt.close()


# noinspection DuplicatedCode
def prep_corpus_analyze(arg) -> int:
    """
    Analyses corpus multilabel data
    ./mulabel prep corpus_analyze -c 'newsmon_sl_p1_s0*.csv' --label_col label
    ./mulabel prep corpus_analyze -c 'mulabel_sl_p1_s0_article*.csv' --label_col labels
    ./mulabel prep corpus_analyze -c 'mulabel_sl_p1_s1_article*.csv' --label_col labels
    ./mulabel prep corpus_analyze -c 'map_articles*.csv' --label_col tags
    ./mulabel prep corpus_analyze -c 'eurlex*.csv' --label_col label
    ./mulabel prep corpus_analyze -c '20news*.csv' --label_col mc_label
    ./mulabel prep corpus_analyze -c 'reuters-21578*.csv' --label_col topics
    """
    file_paths = glob.glob(os.path.join(arg.data_in_dir, arg.collection))
    l_col = arg.label_col

    dfs = []
    for file in file_paths:
        logger.info(f'Reading file %s', file)
        tmp_df = pd.read_csv(file, encoding='utf-8')
        if ptypes.is_string_dtype(tmp_df[l_col]):
            tmp_df[l_col] = tmp_df[l_col].apply(ast.literal_eval)
        elif ptypes.is_integer_dtype(tmp_df[l_col]):
            tmp_df[l_col] = tmp_df[l_col].apply(lambda x: [x])
        dfs.append(tmp_df)

    article_df = pd.concat(dfs, ignore_index=True)
    if 'lang' in article_df:
        logger.info('Got languages in dataset: %s', article_df["lang"].unique())

    num_samples = article_df.shape[0]
    logger.info('Number of samples: %s', num_samples)
    logger.info('With columns: %s', article_df.columns)

    # Create a list of all tags
    all_tags = []
    for tags in article_df[l_col]:
        all_tags.extend(tags)

    # Count the occurrences of each tag
    tag_counts = Counter(all_tags)

    # Construct the tag counts dataframe
    tag_dict = {'tag': [], 'count': []}
    for label, count in tag_counts.items():
        tag_dict['tag'].append(label)
        tag_dict['count'].append(count)

    tag_df = pd.DataFrame(tag_dict).sort_values('tag', ascending=True)

    num_tags = tag_df.shape[0]
    logger.info('Number of labels: %s', num_tags)

    bin_size = 5
    labels_bins = [i for i in range(0, 501, bin_size)]
    labels_bins.append(float('inf'))
    label_histogram_counts = pd.cut(tag_df['count'], bins=labels_bins).value_counts().sort_index()
    print(label_histogram_counts.head())
    label_histogram_percentages = (label_histogram_counts / num_tags) * 100

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    ax = label_histogram_counts.plot(kind='bar', width=0.9)

    # Get the current tick locations and labels
    ticks = ax.get_xticks()
    labels = [item.get_text() for item in ax.get_xticklabels()]

    # Keep only every 10th tick and label
    new_ticks = ticks[::10]
    new_labels = [str(int(tick * bin_size)) if i % 2 == 0 else '' for i, tick in enumerate(new_ticks)]

    # Set the new ticks and labels
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(new_labels)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100.0))  # Adjust base

    # Add a dotted vertical line at the second xtick
    ten_xtick = 1.5
    ax.axvline(x=ten_xtick, color='red', linestyle=':', linewidth=2, zorder=1)

    fifty_xtick = 10.5
    ax.axvline(x=fifty_xtick, color='red', linestyle=':', linewidth=2, zorder=1)

    fiveh_xtick = 100.5
    ax.axvline(x=fiveh_xtick, color='red', linestyle=':', linewidth=2, zorder=1)

    sum_10 = tag_df[tag_df['count'] <= 10]['count'].count()
    ax.annotate(f'{sum_10} labels\n(≤ 10 occurrences)', xy=(ten_xtick, ax.get_ylim()[1]),
                xytext=(10, -60), textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.9),
                fontsize=12,  # font size
                fontweight='bold',  # font weight
                zorder=5
                )
                #arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


    sum_50 = tag_df[tag_df['count'] <= 50]['count'].count()
    ax.annotate(f'{sum_50} labels\n(≤ 50 occurrences)', xy=(fifty_xtick, ax.get_ylim()[1]),
                xytext=(10, -230), textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.9),
                fontsize=12,  # font size
                fontweight='bold',  # font weight
                zorder=5
                )
                #arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    sum_500 = tag_df[tag_df['count'] > 500]['count'].count()
    ax.annotate(f'{sum_500} labels\n(> 500 occurrences)', xy=(fiveh_xtick, ax.get_ylim()[1]),
                xytext=(-145, -360), textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.9),
                fontsize=12,  # font size
                fontweight='bold',  # font weight
                zorder=5
                )
                #arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    #plt.xlabel('Number of occurrences in documents', fonsize=12)
    #plt.ylabel('Number of labels', fonsize=12)
    #plt.title('Histogram of Label Occurrences')
    ax.set_xlabel('Number of occurrences in documents', fontdict={'fontsize': 12})
    ax.set_ylabel('Number of labels', fontdict={'fontsize': 12})
    plt.show()

    tags_diversity = {}
    sum_tags_per_sample = 0
    sum_tags_over_all = 0
    sample_tag_counts = []
    for tags in article_df[l_col]:
        sum_tags_per_sample += len(tags)
        sample_tag_counts.append(len(tags))
        sum_tags_over_all += (len(tags) / num_tags)
        tags_s = str(sorted(set(tags)))
        if not tags_s in tags_diversity:
            tags_diversity[tags_s] = 1
        else:
            tags_diversity[tags_s] += 1

    label_density = sum_tags_over_all / num_samples
    label_cardinality = sum_tags_per_sample / num_samples
    label_diversity = len(tags_diversity)

    logger.info(f'Label density: {label_density}')
    logger.info(f'Label cardinality: {label_cardinality}')
    logger.info(f'Label diversity: {label_diversity}')

    std_dev_cols = np.std(sample_tag_counts, axis=0)
    mean_cols = np.mean(sample_tag_counts, axis=0)
    logger.info(f'Mean {mean_cols} Standard deviation {std_dev_cols}')
    return 0