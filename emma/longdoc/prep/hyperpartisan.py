import os
from typing import Tuple, List

import tqdm
import logging
import jsonlines

from ...core.labels import BinaryLabeler
from .utils import _download_file, _unzip_file, _write_csv

logger = logging.getLogger('longdoc.prep.hyperpartisan')


def read_hyperpartisan_data(hyper_file_path) -> Tuple[List[str], List[str]]:
    """
    Read a jsonl file for Hyperpartisan News Detection data and return lists of documents and labels
    :param hyper_file_path: path to a jsonl file
    :return: lists of documents and labels
    """
    documents = []
    labels = []
    with jsonlines.open(hyper_file_path) as reader:
        for doc in tqdm.tqdm(reader):
            documents.append(doc['text'])
            labels.append(doc['label'])

    return documents, labels


def _prep(dl_dir: str, split_dir: str):
    articles_zip = 'https://zenodo.org/record/1489920/files/articles-training-byarticle-20181122.zip'
    ground_truth_zip = 'https://zenodo.org/record/1489920/files/ground-truth-training-byarticle-20181122.zip'
    hp_splits_json = 'https://raw.githubusercontent.com/allenai/longformer/master/scripts/hp-splits.json'
    hp_preprocess_py = 'https://raw.githubusercontent.com/allenai/longformer/master/scripts/hp_preprocess.py'

    # Download and unzip articles and ground truth
    logger.info('Downloading')
    _download_file(articles_zip, dl_dir)
    _download_file(ground_truth_zip, dl_dir)
    logger.info('Extracting')
    _unzip_file(os.path.join(dl_dir, articles_zip.split('/')[-1]), dl_dir)
    _unzip_file(os.path.join(dl_dir, ground_truth_zip.split('/')[-1]), dl_dir)

    # Download additional files
    _download_file(hp_splits_json, dl_dir)
    _download_file(hp_preprocess_py, dl_dir)
    logger.info('Processing')
    # Execute the preprocessing script
    curr_path = os.curdir
    os.chdir(dl_dir)
    os.system(f'python hp_preprocess.py --output-dir {split_dir}')
    os.chdir(curr_path)

    text_set = {}
    label_set = {}
    enc = BinaryLabeler()
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(split_dir, split + '.jsonl')
        text_set[split], label_set[split] = read_hyperpartisan_data(file_path)
        enc.extend(label_set[split])

    enc.fit()
    vectorized_labels = {}
    for split in ['train', 'dev', 'test']:
        vectorized_labels[split] = enc.vectorize(label_set[split])

    for split in ['train', 'dev', 'test']:
        _write_csv(
            text_set[split], label_set[split], vectorized_labels[split], os.path.join(split_dir, split + '.csv')
        )
        os.remove(os.path.join(split_dir, split + '.jsonl'))
    logger.info('Finished')
