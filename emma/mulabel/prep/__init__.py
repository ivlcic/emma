import os
import json
import csv
import logging
from argparse import ArgumentParser

from transformers import AutoTokenizer
from weaviate.util import generate_uuid5

from ..tokenizer import get_segmenter
from ..utils import compute_arg_collection_name, load_map_file, construct_span_contexts, write_map_file, \
    __supported_languages
from ...core.args import CommonArguments

logger = logging.getLogger('mulabel.prep')


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


def prep_corpus(arg) -> int:
    """
    ./mulabel prep corpus -c Label -l sl --public --seed_only --postfix 2023_01
    ./mulabel prep corpus -c Label -l sl,sr --public --seed_only --postfix 2023_01
    """
    logger.debug("Starting data gathering to simplify format.")

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

    article_map_file_name = os.path.join(arg.data_in_dir, f'map_articles_{arg.postfix}.csv')
    article_cols = [
        'uuid', 'public', 'created', 'published', 'country', 'mon_country', 'lang', 'script', 'm_id',
        'rel_path', 'url', 'sent', 'words', 'sp_tokens', 'tags_count', 'tags'
    ]
    map_articles = load_map_file(article_map_file_name, article_cols)

    segmenters = {}
    for lang in arg.lang:
        segmenters[lang] = get_segmenter(lang, arg.data_out_dir)

    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')

    file_name = os.path.join(arg.data_in_dir, f'data_{arg.postfix}.jsonl')
    with open(file_name, 'r', encoding='utf8') as json_file:
        json_data = json.load(json_file)

    a_file_name = os.path.join(arg.data_out_dir, f'{arg.collection}_article_{arg.postfix}.csv')
    l_file_name = os.path.join(arg.data_out_dir, f'{arg.collection}_labels_{arg.postfix}.csv')
    with open(l_file_name, 'w', encoding='utf8') as l_file, \
            open(a_file_name, 'w', encoding='utf8') as a_file:
        a_writer = csv.writer(a_file)
        a_writer.writerow(['id', 'date', 'm_id', 'public', 'n_tokens', 'text', 'labels'])
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
            if article_idx < arg.max_records:
                break

    write_map_file(
        maps['trained_labels'],
        os.path.join(arg.data_out_dir, f'{arg.collection}_map_labels_{arg.postfix}.csv'),
        label_cols
    )
    return 0
