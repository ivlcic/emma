import os
import glob
import tqdm
import json
import logging

from ...core.labels import MultilabelLabeler
from .utils import _download_file, _unzip_file, _remove_directory, _move_files, _write_csv

logger = logging.getLogger('longdoc.prep.eurlex')


def read_eurlex_file(eur_file_path):
    tags = []
    with open(eur_file_path) as file:
        data = json.load(file)

    inverted_sections = []
    inverted_sections.extend(data['main_body'])
    inverted_sections.append(data['recitals'])
    inverted_sections.append(data['header'])

    sections = [data['header'], data['recitals']]
    sections.extend(data['main_body'])

    for concept in data['concepts']:
        tags.append(concept)

    return '\n'.join(sections), '\n'.join(inverted_sections), tags


def _prep(dl_dir: str, split_dir: str) -> None:
    datasets_zip = 'http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip'
    logger.info('Downloading')
    _download_file(datasets_zip, dl_dir)
    logger.info('Extracting')
    _unzip_file(os.path.join(dl_dir, datasets_zip.split('/')[-1]), dl_dir)
    logger.info('Processing')
    _remove_directory(os.path.join(dl_dir, "__MACOSX"))
    dataset_inner_directory = os.path.join(dl_dir, "dataset")
    if os.path.exists(dataset_inner_directory):
        _move_files(dataset_inner_directory, split_dir)
        _remove_directory(dataset_inner_directory)

    text_set = {'train': [], 'dev': [], 'test': []}
    inverted_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}
    labeler = MultilabelLabeler()
    for split in ['train', 'dev', 'test']:
        file_paths = glob.glob(os.path.join(split_dir, split, '*.json'))
        for file_path in tqdm.tqdm(sorted(file_paths)):
            text, inverted_text, tags = read_eurlex_file(file_path)
            text_set[split].append(text)
            inverted_set[split].append(inverted_text)
            label_set[split].append(tags)
            os.remove(file_path)
        labeler.collect(label_set[split])
        os.rmdir(os.path.join(split_dir, split))

    labeler.fit()
    vectorized_labels = {}
    for split in ['train', 'dev', 'test']:
        vectorized_labels[split] = labeler.vectorize(label_set[split])
        _write_csv(
            text_set[split], label_set[split], os.path.join(split_dir, split + '.csv'), 'ml_label'
        )
    base_name = os.path.basename(split_dir)
    parent_dir = os.path.dirname(split_dir)
    inverted_dir = os.path.join(parent_dir, base_name + 'inv')
    if not os.path.exists(inverted_dir):
        os.makedirs(inverted_dir)

    for split in ['train', 'dev', 'test']:
        _write_csv(
            inverted_set[split], label_set[split], os.path.join(inverted_dir, split + '.csv'), 'ml_label'
        )
    logger.info('Finished')
