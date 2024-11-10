import ast
import glob
import os
import json
import csv
import logging
import uuid
import uuid as uuid_lib
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from argparse import ArgumentParser
from collections import Counter

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

from ..tokenizer import get_segmenter
from ..utils import compute_arg_collection_name, load_map_file, construct_span_contexts, write_map_file, \
    load_add_corpus_part, __supported_languages
from ...core.args import CommonArguments

logger = logging.getLogger('mulabel.prep')

__label_columns = ['label', 'tags', 'ml_label', 'mc_label', 'topics']

__social_media = {
    '8e3b359f', '3e1c137d', '86f18af6', '1fd92aa0', 'c0953029', '1843f51e',
    '151a2b9a', '05b54365', '0e9d50b8', '9f6a5e6c', 'f789b185'
}


# noinspection DuplicatedCode
def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_in_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-i', '--data_out_dir'))
    CommonArguments.split_data_dir(module_name, parser, ('-s', '--data_split_dir'))
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
        '--label_col',
        help=f'Use label column.'
             f'You can use a comma separated list of {__label_columns}',
        type=str, choices=__label_columns, default='label'
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
        '--postfix', help='Read/write data file postfix.', type=str, default='2023_01'
    )
    parser.add_argument(
        '--max_records', help='Maximum number of records to read.', type=int, default=-1
    )


def prep_corpus_extract(arg) -> int:
    """
    Extracts and preprocesses corpus data (aligns labels, finds relevant parts, filters data etc...).
    Produces two files tmp/mulabel/collection_name.csv and tmp/mulabel/lrp_collection_name.csv.
    The second file has label relevant passages while the first one has the complete texts.
    ./mulabel prep corpus_extract -l sl --public --seed_only --postfix 2023_01
    ./mulabel prep corpus_extract -c Label -l sl --public --seed_only --postfix 2023_01
    ./mulabel prep corpus_extract -c Label -l sl,sr --public --seed_only --postfix 2023_01
    """
    logger.debug("Starting extracting data to simplify format.")
    os.environ['HF_HOME'] = arg.data_out_dir  # local tmp dir

    compute_arg_collection_name(arg)
    label_cols = ['name', 'count', 'parent_id', 'monitoring_country', 'monitoring_industry']
    maps = {
        'kwes': load_map_file(
            os.path.join(arg.data_in_dir, 'map', 'map_kwe_tags.csv'), ['tag_id', 'expr']
        ),
        'labels': load_map_file(
            os.path.join(arg.data_in_dir, 'map', 'map_tags.csv'), label_cols
        ),
        'seed_labels': load_map_file(
            os.path.join(arg.data_in_dir, 'map', 'map_seed_tags.csv'), label_cols
        ),
        'trained_labels': {}
    }

    segmenters = {}
    for lang in arg.lang:
        segmenters[lang] = get_segmenter(lang, arg.data_out_dir)

    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')
    if ',' in arg.postfix:
        arg.postfix = arg.postfix.split(',')
    else:
        arg.postfix = [arg.postfix]

    article_input_cols = [
        'uuid', 'public', 'created', 'published', 'country', 'mon_country', 'lang', 'script', 'm_id',
        'rel_path', 'url', 'sent', 'words', 'sp_tokens', 'tags_count', 'tags'
    ]

    article_dup_cols = [
        'a_uuid', 'date', 'text', 'similar_uuid', 'similar_id'
    ]

    article_output_cols = [
        'a_id', 'a_uuid', 'date', 'm_id', 'public', 'lang', 'n_tokens',
        'text', 'label', 'label_info', 'm_social', 'dup'
    ]

    duplicate_file_name = os.path.join(
        arg.data_in_dir, f'{arg.collection}_duplicates.csv'
    )
    duplicates = {}
    if os.path.exists(duplicate_file_name):
        duplicates = load_map_file(duplicate_file_name, article_dup_cols)

    for postfix in arg.postfix:
        article_map_file_name = os.path.join(arg.data_in_dir, 'map', f'map_articles_{postfix}.csv')

        map_articles = load_map_file(article_map_file_name, article_input_cols)

        file_name = os.path.join(arg.data_in_dir, 'src', f'data_{postfix}.jsonl')
        with open(file_name, 'r', encoding='utf8') as json_file:
            json_data = json.load(json_file)

        a_file_name = os.path.join(arg.data_out_dir, f'{arg.collection}_{postfix}.csv')
        l_file_name = os.path.join(arg.data_out_dir, f'lrp_{arg.collection}_{postfix}.csv')
        with open(l_file_name, 'w', encoding='utf8') as l_file, \
                open(a_file_name, 'w', encoding='utf8') as a_file:
            a_writer = csv.writer(a_file)
            a_writer.writerow(article_output_cols)
            l_writer = csv.writer(l_file)
            l_header_written = False
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

                segmenter = segmenters[article['lang']]
                if not segmenter:
                    logger.warning('Missing [%s] tokenizer', article['lang'])
                    continue

                article['m_social'] = 0
                if article['m_id'] in __social_media:
                    article['m_social'] = 1

                article['dup'] = 0
                if article['id'] in duplicates:
                    article['dup'] = 1

                labels = []
                label_ids = []
                for tag in article['tags']:
                    labels.append(
                        {'id': tag['id'], 'name': maps['labels'][tag['id']]['name']}
                    )
                    label_ids.append(tag['id'])
                    if not tag['id'] in maps['trained_labels']:
                        maps['trained_labels'][tag['id']] = maps['labels'][tag['id']]

                text, db_labels = construct_span_contexts(
                    article, segmenter, maps, [1, 3, 5, 7, 9]
                )
                n_tokens = 0
                if text:
                    tokens = tokenizer.tokenize(text)
                    n_tokens = len(tokens)

                a_writer.writerow(
                    [
                        article['id'],
                        article['uuid'],
                        article['date'],
                        article['m_id'],
                        article['public'],
                        article['lang'],
                        n_tokens,
                        text,
                        label_ids,
                        labels,
                        article['m_social'],
                        article['dup']
                    ]
                )

                for db_label in db_labels:
                    db_label['uuid'] = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, str(db_label)))
                    if db_label['uuid'] in stored:
                        continue

                    db_label['n_tokens'] = 0
                    if text:
                        tokens = tokenizer.tokenize(db_label['text'])
                        db_label['n_tokens'] = len(tokens)

                    if article_idx % 1000 == 0:
                        logger.info('Processing label [%s]', db_label)

                    if not l_header_written:
                        l_writer.writerow(db_label.keys())
                        l_header_written = True
                    l_writer.writerow(db_label.values())

                    stored.add(db_label['uuid'])
                if 0 < arg.max_records < article_idx:
                    break

        write_map_file(
            maps['trained_labels'],
            os.path.join(arg.data_out_dir, f'{arg.collection}_{postfix}_map_labels.csv'),
            label_cols
        )
    return 0


def prep_corpus_merge(arg) -> int:
    """
    Merges parts of a corpus based on a postfix(es) and removes single occurring label samples
    and duplicates (if marked as such).
    ./mulabel prep corpus_merge -l sl --public --postfix 2023_01,2023_02,2023_03,2023_04,2023_05
    ./mulabel prep corpus_merge -l sl --public --seed_only --postfix 2023_01,2023_02,2023_03,2023_04,2023_05
    ./mulabel prep corpus_merge -l sl,sr --public --seed_only --postfix 2023_01,2023_02,2023_03,2023_04,2023_05
    """
    logger.debug("Starting preparing data for training to simplify format.")
    os.environ['HF_HOME'] = arg.data_out_dir  # local tmp dir
    compute_arg_collection_name(arg)

    if ',' in arg.postfix:
        arg.postfix = arg.postfix.split(',')
    else:
        arg.postfix = [arg.postfix]

    l_col = arg.label_col

    labels_df = pd.DataFrame()
    lrp_df = pd.DataFrame()
    data_df = pd.DataFrame()
    for postfix in arg.postfix:
        l_file_name = os.path.join(arg.data_out_dir, f'{arg.collection}_{postfix}_map_labels.csv')
        temp_df = pd.read_csv(l_file_name, encoding='utf-8')
        labels_df = pd.concat([labels_df, temp_df])

        lrp_file_name = os.path.join(arg.data_out_dir, f'lrp_{arg.collection}_{postfix}.csv')
        lrp_df = load_add_corpus_part(lrp_file_name, l_col, lrp_df)

        a_file_name = os.path.join(arg.data_out_dir, f'{arg.collection}_{postfix}.csv')
        data_df = load_add_corpus_part(a_file_name, l_col, data_df)

    if 'dup' in data_df.columns:
        logger.debug("Will drop duplicates (if dup mark exists).")
        data_df = data_df[data_df['dup'] == 0]

    if 'dup' in data_df.columns:
        logger.debug("Will drop lrp duplicates (if dup mark exists).")
        lrp_df = lrp_df[lrp_df['dup'] == 0]

    print(data_df.head())

    # recount labels because original labels were off
    labels_df = labels_df.drop_duplicates(subset='id')
    labels_df = labels_df.drop(columns=['count'])  # drop wrong counts
    logger.info('Label samples:')
    print(labels_df.head())

    # Create a list of all tags
    all_tags = []
    for tags in data_df[l_col]:
        all_tags.extend(tags)

    # Count the occurrences of each tag
    tag_counts = Counter(all_tags)
    # Construct the tag counts dataframe
    count_dict = {'id': [], 'count': []}
    for label, count in tag_counts.items():
        count_dict['id'].append(label)
        count_dict['count'].append(count)
    count_df = pd.DataFrame(count_dict).sort_values('id', ascending=False)
    logger.info('Counted Label Sample 1:')
    print(count_df.head())
    labels_df = pd.merge(labels_df, count_df, on='id', how='inner')
    labels_df = labels_df.sort_values(by='count', ascending=False)
    logger.info('Counted Label Sample 2:')
    print(labels_df.head())

    keep_labels_that_occur = 2  # more or equal than
    # Filter labels based on the count threshold
    valid_labels = labels_df[labels_df['count'] >= keep_labels_that_occur]['id']
    # Function to filter labels in samples
    valid_labels_l = set(valid_labels.tolist())
    print(f'Valid Labels: {len(valid_labels_l)}')

    def filter_labels(label_list):
        return [_label for _label in label_list if _label in valid_labels_l]
    # Apply the filter function to the labels column
    data_df[l_col] = data_df[l_col].apply(filter_labels)
    # Remove samples with no labels
    data_df = data_df[data_df[l_col].map(len) > 0]

    labels_df = labels_df[labels_df['id'].isin(valid_labels)]
    lrp_df = lrp_df[lrp_df['a_id'].isin(data_df['a_id'])]

    # write to disk
    data_df.to_csv(os.path.join(arg.data_in_dir, f'{arg.collection}.csv'), index=False, encoding='utf-8')
    labels_df.to_csv(os.path.join(arg.data_in_dir, f'{arg.collection}_map_labels.csv'), index=False, encoding='utf-8')
    lrp_df.to_csv(os.path.join(arg.data_in_dir, f'lrp_{arg.collection}.csv'), index=False)
    return 0


def prep_corpus_split(arg) -> int:
    coll_name = arg.collection
    compute_arg_collection_name(arg)
    a_file_name = os.path.join(arg.data_in_dir, f'{arg.collection}.csv')
    if not os.path.exists(a_file_name):  # for other data
        a_file_name = os.path.join(arg.data_in_dir, f'{coll_name}.csv')
        data_df = pd.read_csv(a_file_name, encoding='utf-8')
        if 'a_id' not in data_df.columns:
            # synthetic a_id and a_uuid
            data_df['a_uuid'] = [uuid.uuid5(uuid.NAMESPACE_DNS, str(i)) for i in range(len(data_df))]
            data_df['a_id'] = [str(uuid_val).split('-')[0] for uuid_val in data_df['a_uuid']]
        if 'label_col' in arg and arg.label_col and arg.label_col in data_df.columns:
            data_df = data_df.rename(columns={arg.label_col: 'label'})
    else:
        data_df = pd.read_csv(a_file_name, encoding='utf-8')
    if ptypes.is_string_dtype(data_df['label']):
        data_df['label'] = data_df['label'].apply(ast.literal_eval)
    elif ptypes.is_integer_dtype(data_df['label']):
        data_df['label'] = data_df['label'].apply(lambda x: [x])

    lrp_file_name = os.path.join(arg.data_in_dir, f'lrp_{arg.collection}.csv')
    lrp_df = None
    if os.path.exists(lrp_file_name):  # for other data
        lrp_df = pd.read_csv(lrp_file_name, encoding='utf-8')

    # Perform an initial stratified split to create train and temp (dev+test) sets
    train_ids, temp_ids = train_test_split(
        data_df['a_id'], test_size=0.2, random_state=2611
    )

    train_df = data_df[data_df['a_id'].isin(train_ids)]
    dev_test_df = data_df[data_df['a_id'].isin(temp_ids)]

    logger.info('Done with train temp split.')

    # Further split the temp set into dev and test sets
    dev_ids, test_ids = train_test_split(
        dev_test_df['a_id'], test_size=0.5, random_state=2611
    )

    dev_df = dev_test_df[dev_test_df['a_id'].isin(dev_ids)]
    test_df = dev_test_df[dev_test_df['a_id'].isin(test_ids)]

    logger.info('Done with train dev test split.')

    # Convert back to DataFrame for easier manipulation
    if lrp_df is not None:
        lrp_train_df = lrp_df[lrp_df['a_id'].isin(train_df['a_id'])]
        lrp_dev_df   = lrp_df[lrp_df['a_id'].isin(dev_df['a_id'])]
        lrp_test_df  = lrp_df[lrp_df['a_id'].isin(test_df['a_id'])]
        lrp_train_df.to_csv(os.path.join(arg.data_split_dir, f'lrp_{arg.collection}_train.csv'), index=False)
        lrp_dev_df.to_csv(os.path.join(arg.data_split_dir, f'lrp_{arg.collection}_dev.csv'), index=False)
        lrp_test_df.to_csv(os.path.join(arg.data_split_dir, f'lrp_{arg.collection}_test.csv'), index=False)
        logger.info(
            'Number of samples: %s/%s/%s', lrp_train_df.shape[0], lrp_dev_df.shape[0], lrp_test_df.shape[0]
        )

    # Save the splits to CSV files
    train_df.to_csv(os.path.join(arg.data_split_dir, f'{arg.collection}_train.csv'), index=False)
    dev_df.to_csv(os.path.join(arg.data_split_dir, f'{arg.collection}_dev.csv'), index=False)
    test_df.to_csv(os.path.join(arg.data_split_dir, f'{arg.collection}_test.csv'), index=False)
    logger.info('Number of samples: %s/%s/%s', train_df.shape[0], dev_df.shape[0], test_df.shape[0])

    all_labels = []
    for tags in data_df['label']:
        all_labels.extend(tags)
    # Count the occurrences of each tag
    label_counts = Counter(all_labels)
    # Construct the tag counts dataframe
    label_dict = {'label': [], 'count': []}
    for label, count in label_counts.items():
        label_dict['label'].append(label)
        label_dict['count'].append(count)

    labels_df = pd.DataFrame(label_dict).sort_values('label', ascending=True)

    logger.info('Number of labels: %s', labels_df.shape[0])
    labels_df.to_csv(os.path.join(arg.data_split_dir, f'{arg.collection}_labels.csv'), index=False)
    return 0


def prep_corpus_train_tmp(arg) -> int:
    """
    Testing preparing training data.
    ./mulabel prep corpus_train_tmp -c 'eurlex*.csv' --label_col ml_label
    """
    from sklearn.model_selection import train_test_split
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

    data_df = pd.concat(dfs, ignore_index=True)

    t_col = 'text'

    # Convert labels to a binary matrix
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data_df[l_col])
    logger.info('Done with mlb transform.')

    # Perform an initial stratified split to create train and temp (dev+test) sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        data_df[t_col], y, test_size=0.2, stratify=y, random_state=2611
    )
    logger.info('Done with mlb split 1.')
    # Further split the temp set into dev and test sets
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=2611
    )
    logger.info('Done with mlb split 2.')

    # Convert back to DataFrame for easier manipulation
    train_df = pd.DataFrame({t_col: X_train, l_col: mlb.inverse_transform(y_train)})
    dev_df = pd.DataFrame({t_col: X_dev, l_col: mlb.inverse_transform(y_dev)})
    test_df = pd.DataFrame({t_col: X_test, l_col: mlb.inverse_transform(y_test)})

    # Ensure each label is present in dev and test sets
    def ensure_label_presence(df, labels):
        label_counts = data_df[l_col].explode().value_counts()
        missing_labels = set(labels) - set(label_counts.index)
        return missing_labels

    all_labels = set(mlb.classes_)
    missing_dev_labels = ensure_label_presence(dev_df, all_labels)
    missing_test_labels = ensure_label_presence(test_df, all_labels)

    # Adjust dev and test sets to include missing labels
    for label in missing_dev_labels:
        sample = train_df[train_df[l_col].apply(lambda x: label in x)].iloc[0]
        dev_df = dev_df.append(sample)
        train_df = train_df.drop(sample.name)

    for label in missing_test_labels:
        sample = train_df[train_df[l_col].apply(lambda x: label in x)].iloc[0]
        test_df = test_df.append(sample)
        train_df = train_df.drop(sample.name)

    # Save the splits to CSV files
    f_name = os.path.join(arg.data_in_dir, f'{arg.collection}_a_train.csv')
    train_df.to_csv(f_name, index=False, encoding='utf-8')
    f_name = os.path.join(arg.data_in_dir, f'{arg.collection}_a_dev.csv')
    dev_df.to_csv(f_name, index=False, encoding='utf-8')
    f_name = os.path.join(arg.data_in_dir, f'{arg.collection}_a_test.csv')
    test_df.to_csv(f_name, index=False, encoding='utf-8')
    return 0


def prep_corpus_eurlex(arg) -> int:
    return 0


# noinspection DuplicatedCode
def prep_corpus_analyze(arg) -> int:
    """
    Analyses corpus multilabel data
    ./mulabel prep corpus_analyze -c 'mulabel_sl_p1_s0_article*.csv' --label_col labels
    ./mulabel prep corpus_analyze -c 'mulabel_sl_p1_s1_article*.csv' --label_col labels
    ./mulabel prep corpus_analyze -c 'map_articles*.csv' --label_col tags
    ./mulabel prep corpus_analyze -c 'eurlex*.csv' --label_col ml_label
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
    ax.axvline(x=ten_xtick, color='red', linestyle=':', linewidth=2)

    sum_10 = tag_df[tag_df['count'] <= 10]['count'].count()
    ax.annotate(f'{sum_10} labels\n(≤ 10 occurrences)', xy=(ten_xtick, ax.get_ylim()[1]),
                xytext=(10, -30), textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),)
                #arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    fifty_xtick = 10.5
    ax.axvline(x=fifty_xtick, color='red', linestyle=':', linewidth=2)
    sum_50 = tag_df[tag_df['count'] <= 50]['count'].count()
    ax.annotate(f'{sum_50} labels\n(≤ 50 occurrences)', xy=(fifty_xtick, ax.get_ylim()[1]),
                xytext=(10, -200), textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),)
                #arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    fiveh_xtick = 100.5
    ax.axvline(x=fiveh_xtick, color='red', linestyle=':', linewidth=2)
    sum_500 = tag_df[tag_df['count'] > 500]['count'].count()
    ax.annotate(f'{sum_500} labels\n(> 500 occurrences)', xy=(fiveh_xtick, ax.get_ylim()[1]),
                xytext=(-115, -370), textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),)
                #arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.xlabel('Number of occurrences in documents')
    plt.ylabel('Number of labels')
    #plt.title('Histogram of Label Occurrences')
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
