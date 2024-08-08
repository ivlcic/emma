import ast
import glob
import os
import json
import csv
import logging
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from collections import Counter

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from weaviate.util import generate_uuid5

from ..tokenizer import get_segmenter
from ..utils import compute_arg_collection_name, load_map_file, construct_span_contexts, write_map_file, \
    __supported_languages
from ...core.args import CommonArguments

logger = logging.getLogger('mulabel.prep')

__label_columns = ['labels', 'tags', 'ml_label', 'mc_label', 'topics']


# noinspection DuplicatedCode
def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_in_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-i', '--data_out_dir'))
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
        type=str, choices=__label_columns, default='labels'
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
    ./mulabel prep corpus_extract -l sl --public --seed_only --postfix 2023_01
    ./mulabel prep corpus_extract -c Label -l sl --public --seed_only --postfix 2023_01
    ./mulabel prep corpus_extract -c Label -l sl,sr --public --seed_only --postfix 2023_01
    """
    logger.debug("Starting extracting data to simplify format.")

    compute_arg_collection_name(arg)
    label_cols = ['name', 'count', 'parent_id', 'monitoring_country', 'monitoring_industry']
    maps = {
        'kwes': load_map_file(
            os.path.join(arg.data_in_dir, 'map_kwe_tags.csv'), ['tag_id', 'expr']
        ),
        'labels': load_map_file(
            os.path.join(arg.data_in_dir, 'map_tags.csv'), label_cols
        ),
        'seed_labels': load_map_file(
            os.path.join(arg.data_in_dir, 'map_seed_tags.csv'), label_cols
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

    for postfix in arg.postfix:
        article_map_file_name = os.path.join(arg.data_in_dir, f'map_articles_{postfix}.csv')
        article_cols = [
            'uuid', 'public', 'created', 'published', 'country', 'mon_country', 'lang', 'script', 'm_id',
            'rel_path', 'url', 'sent', 'words', 'sp_tokens', 'tags_count', 'tags'
        ]
        map_articles = load_map_file(article_map_file_name, article_cols)

        file_name = os.path.join(arg.data_in_dir, f'data_{postfix}.jsonl')
        with open(file_name, 'r', encoding='utf8') as json_file:
            json_data = json.load(json_file)

        a_file_name = os.path.join(arg.data_out_dir, f'{arg.collection}_article_{postfix}.csv')
        l_file_name = os.path.join(arg.data_out_dir, f'{arg.collection}_labels_{postfix}.csv')
        with open(l_file_name, 'w', encoding='utf8') as l_file, \
                open(a_file_name, 'w', encoding='utf8') as a_file:
            a_writer = csv.writer(a_file)
            a_writer.writerow(['id', 'date', 'm_id', 'public', 'lang', 'n_tokens', 'text', 'labels'])
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

                tag_ids = []
                for tag in article['tags']:
                    tag_ids.append(tag['id'])
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
                        article['date'],
                        article['m_id'],
                        article['public'],
                        article['lang'],
                        n_tokens,
                        text,
                        tag_ids,
                    ]
                )

                for db_label in db_labels:
                    db_label['uuid'] = generate_uuid5(db_label)
                    if db_label['uuid'] in stored:
                        continue

                    db_label['n_tokens'] = 0
                    if text:
                        tokens = tokenizer.tokenize(db_label['passage'])
                        db_label['n_tokens'] = len(tokens)

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
            os.path.join(arg.data_out_dir, f'{arg.collection}_map_labels_{postfix}.csv'),
            label_cols
        )
    return 0


def prep_corpus_train(arg) -> int:
    """
    Preps training data. Selects part of a corpus based on a postfix(es) and makes train/eval/test split.
    ./mulabel prep corpus_train -l sl --public --postfix 2023_01,2023_02,2023_03,2023_04,2023_05
    ./mulabel prep corpus_train -l sl --public --seed_only --postfix 2023_01,2023_02,2023_03,2023_04,2023_05
    ./mulabel prep corpus_train -l sl,sr --public --seed_only --postfix 2023_01,2023_02,2023_03,2023_04,2023_05
    """
    logger.debug("Starting preparing data for training to simplify format.")

    compute_arg_collection_name(arg)

    if ',' in arg.postfix:
        arg.postfix = arg.postfix.split(',')
    else:
        arg.postfix = [arg.postfix]

    l_col = arg.label_col

    labels_df = pd.DataFrame()
    data_df = pd.DataFrame()
    for postfix in arg.postfix:
        l_file_name = os.path.join(arg.data_out_dir, f'{arg.collection}_map_labels_{postfix}.csv')
        temp_df = pd.read_csv(l_file_name, encoding='utf-8')
        labels_df = pd.concat([labels_df, temp_df])

        a_file_name = os.path.join(arg.data_out_dir, f'{arg.collection}_article_{postfix}.csv')
        logger.info(f'Reading file %s', a_file_name)
        temp_df = pd.read_csv(a_file_name, encoding='utf-8')
        if ptypes.is_string_dtype(temp_df[l_col]):
            temp_df[l_col] = temp_df[l_col].apply(ast.literal_eval)
        elif ptypes.is_integer_dtype(temp_df[l_col]):
            temp_df[l_col] = temp_df[l_col].apply(lambda x: [x])
        data_df = pd.concat([data_df, temp_df])

    # temp solution:
    if 'lang' not in data_df:
        data_df['lang'] = 'sl'
    logger.info('Article samples:')
    print(data_df.head())
    a_file_name = os.path.join(arg.data_in_dir, f'{arg.collection}_article.csv')
    data_df.to_csv(a_file_name, index=False, encoding='utf-8')

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
    l_file_name = os.path.join(arg.data_in_dir, f'{arg.collection}_map_labels.csv')
    labels_df.to_csv(l_file_name, index=False, encoding='utf-8')

    remove_labels_that_occur = 2
    # Filter labels based on the count threshold
    valid_labels = labels_df[labels_df['count'] >= remove_labels_that_occur]['id']
    # Function to filter labels in samples
    valid_labels_l = set(valid_labels.tolist())
    print(f'Valid Labels: {len(valid_labels_l)}')
    def filter_labels(label_list):
        return [label for label in label_list if label in valid_labels_l]
    # Apply the filter function to the labels column
    data_df[l_col] = data_df[l_col].apply(filter_labels)
    # Remove samples with no labels
    data_df = data_df[data_df[l_col].map(len) > 0]

    a_file_name = os.path.join(arg.data_in_dir, f'{arg.collection}_filtered_article.csv')
    data_df.to_csv(a_file_name, index=False, encoding='utf-8')

    l_file_name = os.path.join(arg.data_in_dir, f'{arg.collection}_filtered_map_labels.csv')
    valid_labels.to_csv(l_file_name, index=False, encoding='utf-8')

    # Perform an initial stratified split to create train and temp (dev+test) sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        data_df['text'], data_df[l_col], test_size=0.2, random_state=2611
    )

    logger.info('Done with train_test_split 1.')

    # Further split the temp set into dev and test sets
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=2611
    )

    logger.info('Done with train_test_split 2.')

    # Convert back to DataFrame for easier manipulation
    train_df = pd.DataFrame({'text': X_train, 'labels': y_train})
    dev_df = pd.DataFrame({'text': X_dev, 'labels': y_dev})
    test_df = pd.DataFrame({'text': X_test, 'labels': y_test})

    # Save the splits to CSV files
    train_df.to_csv(f'{arg.collection}_filtered_article_train.csv', index=False)
    dev_df.to_csv(f'{arg.collection}_filtered_article_dev.csv', index=False)
    test_df.to_csv(f'{arg.collection}_filtered_article_test.csv', index=False)
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

    tag_df = pd.DataFrame(tag_dict).sort_values('tag', ascending=False)
    tag_df.head(10)

    num_tags = tag_df.shape[0]
    logger.info('Number of labels: %s', num_tags)

    ocurr10_bins = [i for i in range(0, 200, 10)]
    ocurr10_histogram_counts = pd.cut(tag_df['count'], bins=ocurr10_bins).value_counts().sort_index()
    ocurr10_histogram_percentages = (ocurr10_histogram_counts / ocurr10_histogram_counts.sum()) * 100

    ocurr100_bins = [i for i in range(0, 3000, 100)]
    ocurr100_histogram_counts = pd.cut(tag_df['count'], bins=ocurr100_bins).value_counts().sort_index()
    ocurr100_histogram_percentages = (ocurr100_histogram_counts / ocurr100_histogram_counts.sum()) * 100

    ocurr1000_bins = [i for i in range(0, 20000, 1000)]
    ocurr1000_histogram_counts = pd.cut(tag_df['count'], bins=ocurr1000_bins).value_counts().sort_index()
    ocurr1000_histogram_percentages = (ocurr1000_histogram_counts / ocurr1000_histogram_counts.sum()) * 100

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))  # 1 row, 2 columns
    ocurr10_histogram_percentages.plot(
        ax=axs[0], kind='bar',
        title='Histogram of Label Occurrences (10 scale)',
        xlabel='Number of occurrences',
        ylabel='Percentage of Labels'
    )
    axs[0].set_xticklabels(ocurr10_bins[1:])

    ocurr100_histogram_percentages.plot(
        ax=axs[1], kind='bar',
        title='Histogram of Label Occurrences (100 scale)',
        xlabel='Number of occurrences',
        ylabel='Percentage of Labels'
    )
    axs[1].set_xticklabels(ocurr100_bins[1:])

    ocurr1000_histogram_percentages.plot(
        ax=axs[2], kind='bar',
        title='Histogram of Label Occurrences (1000 scale)',
        xlabel='Number of occurrences',
        ylabel='Percentage of Labels'
    )
    axs[2].set_xticklabels(ocurr1000_bins[1:])
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
