import ast
import os
import logging

from datetime import datetime
from argparse import ArgumentParser
from typing import Dict, Any, List, Callable, Union

import numpy as np
import pandas as pd
import torch
import faiss

from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
from transformers import AutoModel


from ...core.args import CommonArguments
from ...core.labels import MultilabelLabeler, Labeler
from ...core.metrics import Metrics
from ...core.models import retrieve_model_name_map
from ...core.wandb import initialize_run
from ..tokenizer import get_segmenter
from ..utils import (__supported_languages, __supported_passage_sizes, __label_split_names, __label_splits,
                     compute_arg_collection_name, load_add_corpus_part, parse_arg_passage_sizes,
                     split_csv_by_frequency)

logger = logging.getLogger('mulabel.fa')


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
    parser.add_argument(
        '--test_l_class', type=str, help=f'Test specified label class.',
        choices=['all'].extend(__label_split_names), default='all'
    )


# noinspection DuplicatedCode
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


# noinspection DuplicatedCode
def _init_ebd_models() -> Dict[str, Callable[[Union[str, List[str]]], np.ndarray]]:
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ptm_name, name in retrieve_model_name_map.items():
        if ptm_name == 'bge_m3':

            bmodel = BGEM3FlagModel(
                name, use_fp16=True, device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            def bge_m3_embed(text_to_embed: Union[str, List[str]]):
                return bmodel.encode(text_to_embed)['dense_vecs']

            models[ptm_name] = bge_m3_embed

        if ptm_name == 'jina3':
            jmodel = AutoModel.from_pretrained(
                name, trust_remote_code=True
            )
            jmodel.to(device)

            def jina3_embed(text_to_embed: Union[str, List[str]]):
                return jmodel.encode(text_to_embed, task='text-matching', show_progress_bar=False)

            models[ptm_name] = jina3_embed

    return models


def fa_init_knowledge(args) -> int:
    """
    ./mulabel fa init_knowledge  -c mulabel -l sl --public
    """
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir

    compute_arg_collection_name(args)
    tokenizers = {}
    for lang in args.lang:
        tokenizers[lang] = get_segmenter(lang, args.tmp_dir)

    logger.info(f'Loading models {args.collection}...')
    models = _init_ebd_models()

    logger.info(f'Loading data from {args.collection}...')
    data_as_dicts = _load_data(args, args.collection)

    total_indexed = 0
    indices: Dict[str: Dict[str, Any]] = {}
    for model_name in models:
        indices[model_name] = {}
        indices[model_name]['total_indexed'] = 0
        indices[model_name]['index'] = None

    for chunk in _chunk_data(data_as_dicts, chunk_size=256):
        texts: List[str] = [d['text'] for d in chunk]
        for model_name, model in models.items():
            ret = model(texts)
            batch_size = ret.shape[0]
            dimension = ret.shape[-1]
            if indices[model_name]['total_indexed'] == 0:
                indices[model_name]['index'] = faiss.IndexFlatIP(dimension)
            indices[model_name]['index'].add(ret)
            indices[model_name]['total_indexed'] += batch_size
        #    total_indexed = indices[model_name]['total_indexed']  # last model update is ok for this computation
        #if total_indexed == 512:
        #     break

    index_path = os.path.join(args.data_result_dir, 'index')
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    index_path = os.path.join(args.data_result_dir, 'index', args.collection)
    for model_name in models:
        faiss.write_index(indices[model_name]['index'], index_path + '.' + model_name)

    return 0
