import ast
import os
import re
import csv
import logging
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from typing import List, Dict, Tuple, Union, Optional, Any

from ..core.dataset import TruncatedDataset
from ..core.labels import Labeler, MultilabelLabeler
from .const import __supported_languages, __label_splits, __label_split_names


logger = logging.getLogger('newsmon.utils')


# noinspection DuplicatedCode
def compute_arg_collection_name(arg):
    arg.collection_conf = arg.collection
    if 'lang' in arg and arg.lang:
        arg.lang_conf = arg.lang
        arg.lang = arg.lang.split(',')
        not_in_languages = [lang for lang in arg.lang if lang not in __supported_languages]
        if not_in_languages:
            logger.error(f'Languages {not_in_languages} are not supported!')
            raise ValueError(f'Languages {not_in_languages} are not supported!')
        arg.collection = arg.collection + '_' + '_'.join(arg.lang)
    else:
        arg.lang = __supported_languages
        arg.lang_conf = 'all'
        arg.collection = arg.collection + '_all'
    if 'public' in arg:
        arg.collection = arg.collection + '_p' + ('1' if arg.public else '0')
    if 'seed_only' in arg:
        arg.collection = arg.collection + '_s' + ('1' if arg.seed_only else '0')
    if 'suffix' in arg and arg.suffix:
        arg.collection = arg.collection + '_' + arg.suffix


# noinspection DuplicatedCode
def init_labeler(args) -> Labeler:
    labels_file_name = os.path.join(args.data_in_dir, f'{args.collection}_labels.csv')
    if not os.path.exists(labels_file_name) and 'lrp' in args.collection:
        tmp = re.sub(r'lrp(-\d+)*_', '', args.collection)
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
def construct_datasets(text_set, label_set, tokenizer, max_len: int = 512) -> Tuple[Dict[str, TruncatedDataset], int]:
    datasets = {}
    average_labels_per_sample = 0
    for split in ['dev', 'test', 'train']:
        if split not in text_set.keys():
            continue
        datasets[split] = TruncatedDataset(text_set[split], label_set[split], tokenizer, max_len)
        average_labels_per_sample += datasets[split].average_labels
    average_labels_per_sample /= 3
    avg_k = round(average_labels_per_sample)
    return datasets, avg_k


# noinspection DuplicatedCode
def load_add_corpus_part(f_name: str, l_col: str, data_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    logger.info(f'Reading file %s', f_name)
    temp_df = pd.read_csv(f_name, encoding='utf-8')

    if l_col + '_info' in temp_df.columns and ptypes.is_string_dtype(temp_df[l_col + '_info']):
        temp_df[l_col + '_info'] = temp_df[l_col + '_info'].apply(ast.literal_eval)
    if l_col in temp_df.columns:
        if ptypes.is_string_dtype(temp_df[l_col]):
            temp_df[l_col] = temp_df[l_col].apply(ast.literal_eval)
        elif ptypes.is_integer_dtype(temp_df[l_col]):
            temp_df[l_col] = temp_df[l_col].apply(lambda x: [x])
    if isinstance(data_df, pd.DataFrame) is not None:
        return pd.concat([data_df, temp_df])
    return temp_df


# noinspection DuplicatedCode
def chunk_data(data_list, chunk_size=500):
    """Generator function to yield data in chunks"""
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i + chunk_size]


# noinspection DuplicatedCode
def load_data(arg, coll: str) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    data_file_name = os.path.join(arg.data_in_dir, f'{coll}.csv')
    if not os.path.exists(data_file_name):
        data_file_name = os.path.join(arg.data_out_dir, f'{coll}.csv')
    logger.info(f'Loading data from {data_file_name}...')
    data_df = load_add_corpus_part(data_file_name, 'label')
    if 'lrp' in coll and 'passage_targets' in data_df.columns:
        data_df['passage_targets'] = data_df['passage_targets'].apply(ast.literal_eval)
    return data_df.to_dict(orient='records'), data_df


# noinspection DuplicatedCode
def split_csv_by_frequency(file_path, splits, category_names=None):
    """
    Splits CSV data from a file into dictionaries based on frequency thresholds and custom category names.

    Args:
        file_path (str): The path to the input CSV file.
        splits (list): A list of integers representing frequency thresholds.
                       Example: [10, 500] creates three categories:
                       <=10, between 11 and 499, and >=500.
        category_names (list): A list of strings representing custom names for the categories.
                               Must have one more name than the number of splits.
                               Example: ["Low", "Medium", "High"] for splits [10, 500].

    Returns:
        dict: A dictionary with custom keys (or auto-generated ones) and values as dictionaries of label-count pairs.
    """
    # Sort the splits to ensure correct categorization
    splits = sorted(splits)

    # Validate category_names if provided
    if category_names:
        if len(category_names) != len(splits) + 1:
            raise ValueError("The number of category names must be one more than the number of splits.")
    else:
        # Generate default category names if none are provided
        category_names = [f"<= {splits[0]}"] + \
                         [f"{splits[i] + 1} - {splits[i + 1]}" for i in range(len(splits) - 1)] + \
                         [f">= {splits[-1]}"]

    # Prepare result containers
    categories = {name: {} for name in category_names}

    # Read the CSV file
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)

        # Categorize each row based on the count value
        for row in reader:
            label = row['label']
            count = int(row['count'])

            # Determine which category the count falls into
            if count <= splits[0]:
                categories[category_names[0]][label] = count
            elif count >= splits[-1]:
                categories[category_names[-1]][label] = count
            else:
                for i in range(len(splits) - 1):
                    if splits[i] < count <= splits[i + 1]:
                        categories[category_names[i + 1]][label] = count
                        break

    return categories


def load_labels(split_dir, corpus: str, splits: List[int], names: List[str]) -> Dict[str, Dict[str, int]]:
    l_file_path = os.path.join(split_dir, f'{corpus}_labels.csv')
    if os.path.exists(l_file_path):
        return split_csv_by_frequency(l_file_path, splits, names)
    return {}


# noinspection DuplicatedCode
def filter_metrics(args, labeler: Labeler, y_true: Union[List, np.ndarray], y_prob: Union[List, np.ndarray]):
    if isinstance(y_true, List):
        y_true = np.vstack(y_true, dtype=np.float32)
    if isinstance(y_prob, List):
        y_prob = np.vstack(y_prob, dtype=np.float32)

    target_indices = []
    filter_labels = False
    if args.test_l_class != 'all':
        label_classes = load_labels(args.data_in_dir, args.collection, __label_splits, __label_split_names)
        target_labels = label_classes[args.test_l_class]
        target_indices = [labeler.encoder.classes_.tolist().index(label) for label in target_labels.keys()]
        filter_labels = True

    if not filter_labels:
        return y_true, y_prob

    target_indices = np.array(target_indices)
    # create mask for zeroing non-target labels
    mask = np.zeros(y_true.shape[1], dtype=bool)
    mask[target_indices] = True
    y_true = y_true * mask
    y_prob = y_prob * mask
    # exclude samples with all zeros in y_true
    mask_non_zero = ~np.all(y_true == 0, axis=1)
    y_true = y_true[mask_non_zero]
    y_prob = y_prob[mask_non_zero]

    # keep only target columns
    y_true = y_true[:, target_indices]
    y_prob = y_prob[:, target_indices]
    return y_true, y_prob
