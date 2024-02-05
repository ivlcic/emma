import os
import logging
import torch
import pandas as pd

from argparse import ArgumentParser
from lightning import seed_everything
from torch.utils.data import DataLoader

from ...core.dataset import ChunkDataset
from ...core.labels import Labeler, BinaryLabeler, MultilabelLabeler, MulticlassLabeler
from ...core.args import CommonArguments
from ...core.models import valid_model_names, ModuleFactory, Module

logger = logging.getLogger('longdoc.prep')

corpora = {
    'hyperpartisan', 'eurlex57k', '20news', 'booksummaries'
}


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(module_name, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    CommonArguments.train(parser, 8)
    CommonArguments.num_workers(parser)
    parser.add_argument(
        '--corpus', help=f'Corpora files to train on {corpora}.'
    )
    parser.add_argument(
        '--model_name', type=str, required=True, help=f'Model name: {valid_model_names}'
    )


def _load_data(split_dir, corpus: str):
    text_set = {}
    label_set = {}

    labeler: Labeler = BinaryLabeler()
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(split_dir, corpus, split + '.csv')
        data = pd.read_csv(file_path)
        column_names = data.columns.tolist()
        for col in column_names:
            if 'text' in col:
                text_set[split] = data[col].tolist()
            if 'label' in col:
                if col.startswith('ml_'):
                    labeler = MultilabelLabeler()
                if col.startswith('mc_'):
                    labeler = MulticlassLabeler()
                label_set[split] = data[col].tolist()

    for split in ['train', 'dev', 'test']:
        labeler.collect(label_set[split])
    labeler.fit()
    for split in ['train', 'dev', 'test']:
        label_set[split] = labeler.vectorize(label_set[split])
    return text_set, label_set, labeler


def chunk_collate_fn(batches):
    """
    Create batches for ChunkDataset
    """
    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]


def create_dataloader(module: Module, text_set, label_set, batch_size, num_workers):
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
    for split in ['dev', 'test', 'train']:
        if split not in text_set.keys():
            continue
        dataset = module.dataset_class(text_set[split], label_set[split], module.tokenizer, module.get_max_len())
        if isinstance(dataset, ChunkDataset):
            dataloaders[split] = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                collate_fn=chunk_collate_fn
            )
        else:
            dataloaders[split] = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
            )

    return dataloaders


def _compute_output_name(args):
    scheduler_str = '_warmup' if args.scheduler else ''
    output_model_name = args.model_name + '_' + args.corpus + '_b' + str(args.batch) + \
                        '_e' + str(args.epochs) + '_s' + str(args.seed) + '_lr' + str(args.lr) + scheduler_str
    return output_model_name


def main(args) -> int:
    if args.corpus not in corpora:
        raise RuntimeError(f'Corpus {args.corpus} should be one of {corpora}')

    logger.debug("Starting training")
    seed_everything(args.seed, workers=True)
    logger.debug("Loading data")

    text_set, label_set, labeler = _load_data(args.data_in_dir, args.corpus)
    logger.debug("Loaded data")
    module = ModuleFactory.get_module(
        args.model_name, len(labeler.labels), args.tmp_dir, args.device
    )
    dataloaders = create_dataloader(
        module, text_set, label_set, args.batch, args.num_workers
    )
    output_model_name = _compute_output_name(args)

    logger.debug("Finished training")
    return 0
