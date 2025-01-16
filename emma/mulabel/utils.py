import ast
import os
import csv
import logging

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from typing import List, Dict, Any, Callable, Tuple, Optional, Union

from ..core.dataset import TruncatedDataset
from ..core.labels import Labeler, BinaryLabeler, MultilabelLabeler, MulticlassLabeler

logger = logging.getLogger('mulabel.utils')

__supported_languages = [
    'sl', 'sr', 'sq', 'mk', 'bs', 'hr', 'bg', 'en', 'uk', 'ru',
    'sk', 'cs', 'ro', 'hu', 'pl', 'pt', 'el', 'de', 'es', 'it'
]

__supported_passage_sizes = [1, 3, 5, 7, 9]

__label_splits = [10, 500]
__label_split_names = ['Rare', 'Other', 'Frequent']

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


def parse_arg_passage_sizes(arg):
    if 'passage_sizes' in arg and arg.passage_sizes:
        arg.passage_sizes_conf = arg.passage_sizes
        arg.passage_sizes = ast.literal_eval('[' + arg.passage_sizes + ']')
        not_in_sizes = [int(size) for size in arg.passage_sizes if size not in __supported_passage_sizes]
        if not_in_sizes:
            logger.error(f'Passage sizes {not_in_sizes} are not supported!')
            raise ValueError(f'Passage sizes {not_in_sizes} are not supported!')
    else:
        arg.passage_sizes = __supported_passage_sizes


def load_map_file(file_name: str, cols: List[str]) -> Dict[str, Dict[str, Any]]:
    d = {}
    if not os.path.exists(file_name):
        return d
    with open(file_name, encoding='utf-8') as d_file:
        try:
            d_reader = csv.reader(d_file)
            for row_idx, row in enumerate(d_reader):
                if row_idx == 0:
                    continue
                key = row[0]
                d[key] = {}
                for idx, c in enumerate(cols):
                    d[key][c] = row[idx + 1]
        except Exception as e:
            logger.error("Unable to load CSV kwe map file [%s].", file_name, e)
            exit(1)
    return d


def write_map_file(d: Dict[str, Dict[str, Any]], file_name: str, cols: List[str]) -> None:
    with open(file_name, 'w', encoding='utf8') as kwe_file:
        writer = csv.writer(kwe_file)
        row = ['id']
        for c in cols:
            row.append(c)
        writer.writerow(row)
        for k, v in d.items():
            row = [k]
            for c in cols:
                row.append(v[c])
            writer.writerow(row)


def get_element_index_at(c_idx: int, indices: List[int]) -> Tuple[int, int]:
    sent_idx = -1
    char_idx = -1
    last_idx = len(indices) - 1
    for i, s_char_idx in enumerate(indices):
        e_char_idx = indices[i + 1] if i + 1 <= last_idx else s_char_idx
        if s_char_idx <= c_idx < e_char_idx:
            sent_idx = i
            char_idx = c_idx - s_char_idx
            break

    return sent_idx, char_idx


def sentence_segment(text: str, tokenize: Callable[[str], Any], segments: Dict[str, List[Any]]) -> None:
    doc = tokenize(text)
    curr_idx = 0
    for i, sentence in enumerate([sentence.text for sentence in doc.sentences]):
        segments['sentences'].append(sentence)
        curr_idx = text.index(sentence, curr_idx)
        segments['indices'].append(curr_idx)

    if not segments['sentences']:
        return
    segments['indices'].append(segments['indices'][-1] + len(sentence))


def _add_passage_kwes(target_kwes: List, passage_kwes: List) -> None:
    for passage_kwe in passage_kwes:
        found = False
        for target_kwe in target_kwes:
            if target_kwe['id'] == passage_kwe['id']:
                found = True
                break
        if not found:
            target_kwes.append(passage_kwe)


def _construct_passages(passage_size: int, passage_sizes, seed_label, all_sentences, label_ids_sent_idx):
    tmp_db_labels = []
    seen = {}
    max_passage_size = max((size for size in passage_sizes if size < len(all_sentences)), default=passage_sizes[0])
    passage_targets = [passage_size]
    if passage_size >= max_passage_size:
        passage_targets = [size for size in passage_sizes if size >= max_passage_size]

    for label_id, label_data in label_ids_sent_idx.items():
        # store all sentence indices for a given keyword expression match
        # remove the ones that were included in a passage to minimize redundant overlapping passages
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
            kwes = []
            for j in range(start, end):
                if j in passage_center_sentence_indices:
                    passage_center_sentence_indices.remove(j)
                    _add_passage_kwes(kwes, label_data['s_idx_kwe'][j])
                passage.append(all_sentences[j])

            if len(passage) < passage_size:
                continue
            if len(passage) <= 0:
                continue

            db_span_label = seed_label.copy()
            db_span_label['label'] = [label_id]
            db_span_label['label_info'] = [{'id': label_id, 'name': label_data['title'], 'kwe': kwes}]
            db_span_label['passage_targets'] = passage_targets
            db_span_label['passage_cat'] = passage_size
            db_span_label['text'] = ' '.join(passage)
            if db_span_label['text'] not in seen:
                tmp_db_labels.append(db_span_label)
                seen[db_span_label['text']] = db_span_label
            else:
                # compress same passages adding all labels to it
                db_span_label = seen[db_span_label['text']]
                found = False
                for tmp_data in db_span_label['label_info']:
                    if label_id == tmp_data['id']:
                        tmp_data['kwe'].extend(kwes)
                        found = True
                        break
                if not found:
                    db_span_label['label_info'].append({'id': label_id, 'name': label_data['title'], 'kwe': kwes})
                    db_span_label['label'].append(label_id)

    return tmp_db_labels


def construct_span_contexts(article: Dict[str, Any], tokenize: Callable[[str], Any],
                            maps: Dict[str, Any], passage_sizes: List[int]) -> Tuple[str, List[Dict[str, Any]]]:
    segment_spans = {
        'ts': {'name': 'title', 'sentences': [], 'indices': []},
        'bs': {'name': 'body', 'sentences': [], 'indices': []},
    }
    text = ''
    for seg_name, span_field in segment_spans.items():
        if 'text' in article[span_field['name']]:
            text = article[span_field['name']]['text']
        if text:
            sentence_segment(text, tokenize, span_field)

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
        'a_uuid': article['uuid'] if 'uuid' in article else None,
        'a_id': article['id'],
        'm_id': article['m_id'],
        'm_social': article['m_social'],
        'dup': article['dup'],
        'country': article['country'],
        'lang': article['lang'],
        'passage_targets': passage_sizes,
        'passage_cat': 0,
        'text': text
    }

    label_ids_sent_idx = {}  # we'll use this intermediate dict for relevant passage construction
    for tag in article['tags']:
        segment_names = segment_spans.keys()
        all_spans_empty = all(not tag[key] for key in segment_names)
        db_label = seed_label.copy()
        db_label['label'] = [tag['id']]
        db_label['label_info'] = [{'id': tag['id'], 'name': maps['labels'][tag['id']]['name']}]
        if all_spans_empty:
            logger.info(
                'Article [%s] label [%s::%s] has no spans',
                article['id'], tag['id'], maps['labels'][tag['id']]['name']
            )
            # if multiple labels have no spans we just add label
            found = False
            for tmp_db_label in db_labels:
                if tmp_db_label['passage_cat'] == 0:
                    tmp_db_label['label'].append(tag['id'])
                    tmp_db_label['label_info'].append({'id': tag['id'], 'name': maps['labels'][tag['id']]['name']})
                    found = True
                    break
            if not found:
                db_labels.append(db_label)
            continue

        # init each label to sentences index mapping
        if tag['id'] not in label_ids_sent_idx:
            label_ids_sent_idx[tag['id']] = {'s_idx': [], 'title': '', 's_idx_kwe': {}}

        prev_segment_offset = 0
        for segment_name in segment_names:  # title, body
            sentences = segment_spans[segment_name]['sentences']
            for span in tag[segment_name]:
                # single sentence passage matching keyword expression (1 label <-> N kwe)
                center_sentence_idx, span_sent_idx = get_element_index_at(
                    span['s'], segment_spans[segment_name]['indices']
                )
                if 0 <= center_sentence_idx < len(sentences):
                    true_idx = center_sentence_idx + prev_segment_offset
                    span_sentence = sentences[center_sentence_idx]
                    if true_idx not in label_ids_sent_idx[tag['id']]['s_idx']:
                        label_ids_sent_idx[tag['id']]['s_idx'].append(true_idx)
                        label_ids_sent_idx[tag['id']]['s_idx_kwe'][true_idx] = [
                            {'id': span['kwe'], 'value': span['m']}
                        ]
                    else:
                        label_ids_sent_idx[tag['id']]['s_idx_kwe'][true_idx].append(
                            {'id': span['kwe'], 'value': span['m']}
                        )
                    label_ids_sent_idx[tag['id']]['title'] = maps['labels'][tag['id']]['name']
            prev_segment_offset += len(sentences)

    # add a larger (multi-sentence) passage matching keyword expression
    # considering overlaps also ... hence the complicated code
    all_sentences = [s for sentences in segment_spans.values() for s in sentences['sentences']]
    for passage_size in passage_sizes:
        tmp_db_labels = _construct_passages(passage_size, passage_sizes, seed_label, all_sentences, label_ids_sent_idx)
        db_labels.extend(tmp_db_labels)

    return text, db_labels


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


def compute_model_output_name(args):
    scheduler_str = '_warmup' if args.scheduler else ''
    output_model_name = args.ptm_name + '_' + args.corpus + '_x' + str(args.run_id) + '_b' + str(args.batch)
    output_model_name += '_e' + str(args.epochs) + '_s' + str(args.seed) + '_lr' + str(args.lr)
    if 'seq_len' in args and args.seq_len > 0:
        output_model_name += f'_l{args.seq_len}'
    if 'grad_acc' in args and args.grad_acc > 0:
        output_model_name += f'_ga{args.grad_acc}'
    output_model_name += scheduler_str
    return output_model_name


def load_train_data(split_dir, corpus: str, test_only: bool = False):
    text_set = {}
    label_set = {}

    labeler: Labeler = BinaryLabeler()
    for split in ['train', 'dev', 'test']:
        if test_only and split != 'test':
            continue
        file_path = os.path.join(split_dir, corpus, split + '.csv')
        if not os.path.isfile(file_path):
            file_path = os.path.join(split_dir, corpus + f'_{split}.csv')
            if not os.path.isfile(file_path):
                continue
        data = pd.read_csv(file_path)
        if 'train' in split:  # do shuffle
            data = data.sample(frac=1).reset_index(drop=True)  # seed is set before
        column_names = data.columns.tolist()
        for col in column_names:
            if 'text' in col:
                text_set[split] = data[col].tolist()
            if 'label' in col:
                if ptypes.is_string_dtype(data[col]):
                    value = data[col].iloc[0]
                    if value.startswith('[{'):
                        continue
                    if value.startswith('['):
                        labeler = MultilabelLabeler()
                        data[col] = data[col].apply(lambda x: ast.literal_eval(x))
                if col.startswith('ml_'):
                    labeler = MultilabelLabeler()
                    data['ml_label'] = data['ml_label'].apply(lambda x: ast.literal_eval(x))
                if col.startswith('mc_'):
                    labeler = MulticlassLabeler()
                label_set[split] = data[col].tolist()

    l_file_path = os.path.join(split_dir, f'{corpus}_labels.csv')
    if os.path.exists(l_file_path):
        with open(l_file_path, 'r') as l_file:
            all_labels = [line.split(',')[0].strip() for line in l_file]
            if all_labels[0] == 'label':
                all_labels.pop(0)
            labeler.collect(all_labels)
    else:
        for split in ['train', 'dev', 'test']:
            labeler.collect(label_set[split])
    labeler.fit()
    for split in ['train', 'dev', 'test']:
        if test_only and split != 'test':
            continue
        label_set[split] = labeler.vectorize(label_set[split])
    return text_set, label_set, labeler


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


def init_labeler(args) -> Labeler:
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


def load_labels(split_dir, corpus: str, splits: List[int], names: List[str]) -> Dict[str, Dict[str, int]]:
    l_file_path = os.path.join(split_dir, f'{corpus}_labels.csv')
    if os.path.exists(l_file_path):
        return split_csv_by_frequency(l_file_path, splits, names)


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
