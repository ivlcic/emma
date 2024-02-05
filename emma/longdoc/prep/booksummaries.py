import json
import logging
import os
from typing import Tuple, Any, Dict, Optional

import pandas as pd

from ...core.labels import MultilabelLabeler
from .utils import _download_file, _extract_tgz, _write_csv

logger = logging.getLogger('longdoc.prep.booksummaries')


# noinspection PyBroadException
def _parse_json_column(genre_data: str) -> Optional[Dict[str, Any]]:
    """
    Read genre information as a json string and convert it to a dict
    :param genre_data: genre data to be converted
    :return: dict of genre names
    """
    try:
        return json.loads(genre_data)
    except Exception:
        return None  # when genre information is missing


def _load_booksummaries_data(book_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the Book Summary data and split it into train/dev/test sets
    :param book_path: path to the booksummaries.txt file
    :return: train, dev, test as pandas data frames
    """
    book_df = pd.read_csv(book_path, sep='\t', names=['Wikipedia article ID',
                                                      'Freebase ID',
                                                      'Book title',
                                                      'Author',
                                                      'Publication date',
                                                      'genres',
                                                      'summary'],
                          converters={'genres': _parse_json_column})
    book_df = book_df.dropna(subset=['genres', 'summary'])  # remove rows missing any genres or summaries
    book_df['word_count'] = book_df['summary'].str.split().str.len()
    book_df = book_df[book_df['word_count'] >= 10]
    train = book_df.sample(frac=0.8, random_state=22)
    rest = book_df.drop(train.index)
    dev = rest.sample(frac=0.5, random_state=22)
    test = rest.drop(dev.index)
    return train, dev, test


# noinspection HttpUrlsUsage
def _prep(dl_dir: str, split_dir: str) -> None:
    booksummaries_url = "http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz"
    booksummaries_tar_gz_path = _download_file(booksummaries_url, dl_dir)
    _extract_tgz(booksummaries_tar_gz_path, os.path.dirname(dl_dir))

    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}
    book_path = os.path.join(dl_dir, 'booksummaries.txt')
    train, dev, test = _load_booksummaries_data(book_path)
    text_set['train'] = train['summary'].tolist()
    text_set['dev'] = dev['summary'].tolist()
    text_set['test'] = test['summary'].tolist()

    train_genres = train['genres'].tolist()
    label_set['train'] = [list(genre.values()) for genre in train_genres]
    dev_genres = dev['genres'].tolist()
    label_set['dev'] = [list(genre.values()) for genre in dev_genres]
    test_genres = test['genres'].tolist()
    label_set['test'] = [list(genre.values()) for genre in test_genres]

    labeler = MultilabelLabeler()
    for split in ['train', 'dev', 'test']:
        labeler.collect(label_set[split])
    labeler.fit()

    vectorized_labels = {}
    for split in ['train', 'dev', 'test']:
        vectorized_labels[split] = labeler.vectorize(label_set[split])
        _write_csv(
            text_set[split], label_set[split], os.path.join(split_dir, split + '.csv'), 'ml_label'
        )
