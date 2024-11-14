import os
import ast
from typing import Dict

import torch
import pandas as pd
import pandas.api.types as ptypes

from torch.utils.data import DataLoader

from ...core.dataset import ChunkDataset
from ...core.labels import Labeler, BinaryLabeler, MultilabelLabeler, MulticlassLabeler
from ...core.models import Module


# noinspection DuplicatedCode
def _load_data(split_dir, corpus: str):
    text_set = {}
    label_set = {}

    labeler: Labeler = BinaryLabeler()
    for split in ['train', 'dev', 'test']:
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
        label_set[split] = labeler.vectorize(label_set[split])
    return text_set, label_set, labeler


# noinspection DuplicatedCode
def _get_long_texts_and_labels(text_dict, label_dict, tokenizer, max_length=512):
    """
    Find texts that have more than a given max token length and their labels
    :param text_dict: dict of lists of texts for train/dev/test splits, keys=['train', 'dev', 'test']
    :param label_dict: dict of lists of labels for train/dev/test splits, keys=['train', 'dev', 'test']
    :param tokenizer: tokenizer of choice e.g. LongformerTokenizer, BertTokenizer
    :param max_length: maximum length of sequence e.g. 512
    :return: dicts of lists of texts with more than the max token length and their labels
    """
    long_text_set = {'dev': [], 'test': []}
    long_label_set = {'dev': [], 'test': []}
    for split in ['dev', 'test']:
        long_text_idx = []
        for idx, text in enumerate(text_dict[split]):
            if len(tokenizer.tokenize(text)) > (max_length - 2):
                long_text_idx.append(idx)
        long_text_set[split] = [text_dict[split][i] for i in long_text_idx]
        long_label_set[split] = [label_dict[split][i] for i in long_text_idx]
    return long_text_set, long_label_set


def _chunk_collate_fn(batches):
    """
    Create batches for ChunkDataset
    """
    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]


def _create_dataloader(module: Module, text_set, label_set, batch_size, num_workers) -> Dict[str, DataLoader]:
    """
    Create appropriate dataloaders for the given data
    :param module: module that contains the dataset class, tokenizer and max available text length
    :param text_set: dict of lists of texts for train/dev/test splits, keys=['train', 'dev', 'test']
    :param label_set: dict of lists of labels for train/dev/test splits, keys=['train', 'dev', 'test']
    :param batch_size: batch size for dataloaders
    :param num_workers: number of workers for dataloaders
    :return: set of dataloaders for train/dev/test splits, keys=['train', 'dev', 'test']
    """
    dataloaders = {}
    average_labels_per_sample = 0
    for split in ['dev', 'test', 'train']:
        if split not in text_set.keys():
            continue
        shuffle = False
        if split == 'train':
            shuffle = True
        dataset = module.dataset_class(text_set[split], label_set[split], module.tokenizer, module.get_max_len())
        average_labels_per_sample += dataset.average_labels
        if isinstance(dataset, ChunkDataset):
            dataloaders[split]: DataLoader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                collate_fn=_chunk_collate_fn
            )
        else:
            dataloaders[split] = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
            )
    module.set_average_labels_per_sample(average_labels_per_sample / 3)
    return dataloaders


def _compute_output_name(args):
    scheduler_str = '_warmup' if args.scheduler else ''
    output_model_name = args.ptm_name + '_' + args.corpus + '_x' + str(args.run_id) + '_b' + str(args.batch)
    output_model_name += '_e' + str(args.epochs) + '_s' + str(args.seed) + '_lr' + str(args.lr)
    output_model_name += scheduler_str
    return output_model_name
