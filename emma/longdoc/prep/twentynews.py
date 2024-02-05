import os
import logging
import re

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from ...core.labels import MulticlassLabeler
from .utils import _write_csv

logger = logging.getLogger('longdoc.prep.20newsgroups')


def _clean_20news_data(text_str):
    """
    Clean up 20NewsGroups text data, from CogLTX: https://github.com/Sleepychord/CogLTX/blob/main/20news/process_20news.py
    // SPDX-License-Identifier: MIT
    :param text_str: text string to clean up
    :return: clean text string
    """
    tmp_doc = []
    for words in text_str.split():
        if ':' in words or '@' in words or len(words) > 60:
            pass
        else:
            c = re.sub(r'[>|-]', '', words)
            if len(c) > 0:
                tmp_doc.append(c)
    tmp_doc = ' '.join(tmp_doc)
    tmp_doc = re.sub(r'\([A-Za-z .]*[A-Z][A-Za-z .]*\) ', '', tmp_doc)
    return tmp_doc


def _prep(dl_dir: str, split_dir: str):
    text_set = {}
    label_set = {}
    logger.info('Downloading')
    test_set = fetch_20newsgroups(subset='test', random_state=21)
    train_set = fetch_20newsgroups(subset='train', random_state=21)

    logger.info('Processing')
    text_set['test'] = [_clean_20news_data(text) for text in test_set.data]
    label_set['test'] = test_set.target

    train_text = [_clean_20news_data(text) for text in train_set.data]
    train_label_set = train_set.target

    # take 10% of the train set as the dev set
    text_set['train'], text_set['dev'], label_set['train'], label_set['dev'] = train_test_split(
        train_text, train_label_set, test_size=0.10, random_state=21
    )

    trans_labels = {i: test_set.target_names[i] for i in range(len(test_set.target_names))}

    labeler = MulticlassLabeler()
    for split in ['train', 'dev', 'test']:
        labeler.collect(label_set[split])
    labeler.fit()

    vectorized_labels = {}
    for split in ['train', 'dev', 'test']:
        vectorized_labels[split] = labeler.vectorize(label_set[split])
        _write_csv(
            text_set[split],
            label_set[split],
            os.path.join(split_dir, split + '.csv'),
            'mc_label',
            trans_labels
        )
    logger.info('Finished')
