import logging
import os
import pandas as pd

from argparse import ArgumentParser
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

from transformers import AutoTokenizer

from ...core.walker import Params, State, walk_range
from ...core.args import CommonArguments

logger = logging.getLogger('mulabel.prep')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.tmp_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_out_dir'))
    beginning_of_day = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    parser.add_argument(
        '-s', '--start_date', help='Articles start selection date.', type=str,
        default=beginning_of_day.astimezone(timezone.utc).isoformat()
    )
    next_day = beginning_of_day + timedelta(days=1)
    parser.add_argument(
        '-e', '--end_date', help='Articles end selection date.', type=str,
        default=next_day.astimezone(timezone.utc).isoformat()
    )
    parser.add_argument(
        '-c', '--country', help='Articles selection country.', type=str
    )


def get_parent_labels(arg) -> Dict[str, Any]:
    parent_df = pd.read_csv(os.path.join(arg.data_out_dir, f'label_parent.csv'), encoding='utf-8')
    rows = parent_df.to_dict(orient='records')
    parents = {}
    for index, row in enumerate(rows):
        parents[row['uuid']] = {'country': row['country'], 'name': row['name'], 'c_uuid': row['c_uuid']}
    return parents


def prep_gather_labels(arg) -> int:
    """
    ./mulabel prep gather_labels -i /home/nikola/projects/neuroticla/result/corpus -s 2023-12-02 -e 2023-12-03
    """
    logger.debug("Starting data gathering to simplify formats.")
    params = Params(arg.start_date, arg.end_date, arg.data_in_dir)
    parent_labels = get_parent_labels(arg)
    tags: Dict[str, Dict[str, Any]] = {}
    missing_parents = set()
    num_unknown = {'name': 0}

    def callback(s: State, article: Dict[str, Any]) -> int:
        if 'tags' not in article:
            return 0 ##'52ecd827-90c3-11ee-b653-2fe3bf8a9fb0'

        for t in article['tags']:
            t_uuid = t['uuid']
            t_country = None
            if 'parent' not in t:
                if t_uuid not in parent_labels:
                    missing_parents.add(t_uuid)
                continue
            t_part = t_uuid.split('-')[0]
            if t_part in tags:
                if tags[t_part]['uuid'] != t_uuid:
                    raise RuntimeError(
                        'Tag collision detected for uuid [%s] with uuid [%s]', t_uuid, tags[t_part]
                    )
                else:
                    tags[t_part]['count'] += 1
            else:
                parent_id = t['parent']['uuid']  # t['tags'][0]['refUuid']
                if parent_id in parent_labels:
                    t_country = parent_labels[parent_id]['country']
                if t_country is None:
                    t_country = article['country']['name']
                tag = {
                    'suuid': t_part,
                    'uuid': t_uuid,
                    'country': t_country,
                    'count': 1,
                    'parent': parent_id
                }
                if 'name' in t:
                    tag['name'] = t['name']
                else:
                    tag['name'] = f'Unknown-{num_unknown["name"]}'
                    num_unknown['name'] += 1

                tags[t_part] = tag
        return 1

    walk_range(params, callback)

    # write missing parent labels
    with open(os.path.join(arg.data_out_dir, f'missing_parents.csv'), 'w') as file:
        for item in missing_parents:
            file.write(str(item) + '\n')

    tags_csv = {
        'suuid': [],
        'count': [],
        'country': [],
        'name': [],
        'uuid': [],
        'parent': []
    }
    for k, t in tags.items():
        tags_csv['suuid'].append(t['suuid'])
        tags_csv['count'].append(t['count'])
        tags_csv['country'].append(t['country'])
        tags_csv['name'].append(t['name'])
        tags_csv['uuid'].append(t['uuid'])
        tags_csv['parent'].append(t['parent'])

    df = pd.DataFrame(tags_csv)
    df = df.sort_values('count', ascending=False)
    df.to_csv(os.path.join(arg.data_out_dir, f'tags-{arg.start_date}_{arg.end_date}.csv'), index=False)

    return 0


def append_stats(data: Dict[str, List[Any]], article, title: str, body: str,
                 bert_tokenizer, deb_tokenizer, xlmr_tokenizer) -> None:
    text = title + '\n\n' + body
    tok_s = 0
    sent = 0
    words = 0

    if title:
        tok_s += article['title']['stats']['sp_t']
        sent += article['title']['stats']['sent']
        words += article['title']['stats']['w_t']
    if body:
        tok_s += article['body']['stats']['sp_t']
        sent += article['body']['stats']['sent']
        words += article['body']['stats']['w_t']

    data['sent'].append(sent)
    data['words'].append(words)
    data['tok_b'].append(len(bert_tokenizer.tokenize(text)))
    data['tok_d'].append(len(deb_tokenizer.tokenize(text)))
    data['tok_x'].append(len(xlmr_tokenizer.tokenize(text)))
    data['tok_s'].append(tok_s)
    data['chrs'].append(len(text))


def prep_gather(arg) -> int:
    """
    ./mulabel prep gather -i /home/nikola/projects/neuroticla/result/corpus -s 2023-12-02 -e 2023-12-03
    """
    logger.debug("Starting data gathering to simplify format.")
    params = Params(arg.start_date, arg.end_date, arg.data_in_dir)
    cache_dir = CommonArguments._package_path('tmp', 'prep')
    deb_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/mdeberta-v3-base", cache_dir=cache_dir
    )  # 190M, CC100
    bert_tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-multilingual-cased", cache_dir=cache_dir
    )  # 179M Wikipedia
    xlmr_tokenizer = AutoTokenizer.from_pretrained(
        "FacebookAI/xlm-roberta-base", cache_dir=cache_dir
    )  # 279M, CC100

    data = {
        'suuid': [],
        'published': [],
        'country': [],
        'language': [],
        'type': [],
        'sent': [],
        'words': [],
        'tok_b': [],
        'tok_d': [],
        'tok_x': [],
        'tok_s': [],
        'chrs': [],
        'path': [],
        'label_co': [],
        'n_labels': [],
        'labels': [],
        'title': [],
        'url': []
    }

    collision_tag_uuids: Dict[str, str] = {}
    collision_article_uuids: Dict[str, str] = {}

    def callback(s: State, article: Dict[str, Any]) -> int:
        body = ''
        if 'body' in article and 'text' in article['body'] and len(article['body']['text']) > 0:
            body = article['body']['text'].strip()
        title = ''
        if 'title' in article and 'text' in article['title'] and len(article['title']['text']) > 0:
            title = article['title']['text'].strip().replace('\n', ' ').replace('\r', '')
            data['title'].append(title)
        else:
            data['title'].append('')

        append_stats(data, article, title, body, bert_tokenizer, deb_tokenizer, xlmr_tokenizer)

        a_uuid = article['uuid']
        a_part = a_uuid.split('-')[0]
        if a_part in collision_article_uuids and collision_article_uuids[a_part] != a_uuid:
            raise RuntimeError(
                'Article collision detected for uuid [%s] with uuid [%s]', a_uuid, collision_article_uuids[a_part]
            )

        data['suuid'].append(a_part)
        data['published'].append(article['published'])
        data['country'].append(article['country']['name'])
        if '-' in article['language']:
            data['language'].append(article['language'].split('-')[0])
        else:
            data['language'].append(article['language'])
        if 'media' in article and 'type' in article['media']:
            data['type'].append(article['media']['type']['name'])
        else:
            data['type'].append('unkn')
        data['path'].append(s.rel_path)

        if 'tags' in article:
            tags = set()
            for t in article['tags']:
                t_uuid = t['uuid']
                t_part = t_uuid.split('-')[0]
                if t_part in collision_tag_uuids and collision_tag_uuids[t_part] != t_uuid:
                    raise RuntimeError(
                        'Tag collision detected for uuid [%s] with uuid [%s]', t_uuid, collision_tag_uuids[t_part]
                    )
                tags.add(t_part)
                collision_tag_uuids[t_part] = t_uuid

            data['labels'].append(list(tags))
        else:
            data['labels'].append([])
        data['n_labels'] = len(data['labels'])

        if 'url' in article:
            data['url'].append(article['url'])
        else:
            data['url'].append('')
        logger.debug("%s", article['uuid'])
        collision_article_uuids[a_part] = a_uuid
        return 1

    state = walk_range(params, callback)

    logger.info(
        "Corrected [%s] files [%s::%s] ", state.total, state.start, state.end
    )

    df = pd.DataFrame(data)
    df = df.iloc[::-1]  # reverse sort, because we collect backwards
    df.to_csv(os.path.join(arg.data_out_dir, f'stats-{arg.start_date}_{arg.end_date}.csv'), index=False)

    df = pd.DataFrame(data)
    df = df.iloc[::-1]
    df = df.drop(columns=['title', 'url'])
    df.to_csv(os.path.join(arg.data_out_dir, f'stats-short-{arg.start_date}_{arg.end_date}.csv'), index=False)
    return 0


def prep_analyse(arg) -> int:
    pass
