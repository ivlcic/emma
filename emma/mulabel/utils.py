import os
import csv
import logging

from typing import List, Dict, Any, Callable, Tuple

logger = logging.getLogger('mulabel.utils')

__supported_languages = [
    'sl', 'sr', 'sq', 'mk', 'bs', 'hr', 'bg', 'en', 'uk', 'ru',
    'sk', 'cs', 'ro', 'hu', 'pl', 'pt', 'el', 'de', 'es', 'it'
]


def compute_arg_collection_name(arg):
    arg.collection_conf = arg.collection
    if 'lang' in arg and arg.lang:
        arg.lang = arg.lang.split(',')
        not_in_languages = [lang for lang in arg.lang if lang not in __supported_languages]
        if not_in_languages:
            logger.error('Languages %s are not supported!', [])
        arg.collection = arg.collection + '_' + '_'.join(arg.lang)
    else:
        arg.lang = __supported_languages
        arg.collection = arg.collection + '_all'
    if 'public' in arg:
        arg.collection = arg.collection + '_p' + ('1' if arg.public else '0')
    if 'seed_only' in arg:
        arg.collection = arg.collection + '_s' + ('1' if arg.seed_only else '0')


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
        'country': article['country'],
        'lang': article['lang'],
        'passage_cat': 0,
        'passage': text,
        'kwe': [],
        'kwe_id': []
    }

    label_ids_sent_idx = {}  # we'll use this for relevant passage construction
    all_sentences = []
    for tag in article['tags']:
        segment_names = segment_spans.keys()
        all_spans_empty = all(not tag[key] for key in segment_names)
        db_label = seed_label.copy()
        db_label['label_id'] = tag['id']
        db_label['label_title'] = maps['labels'][tag['id']]['name']
        db_label['kwe'] = []
        db_label['kwe_id'] = []
        if all_spans_empty:
            logger.info(
                'Article [%s] label [%s::%s] has no spans',
                article['id'], tag['id'], maps['labels'][tag['id']]['name']
            )
            db_labels.append(db_label)
            continue

        # init each label to sentences index mapping
        if tag['id'] not in label_ids_sent_idx:
            label_ids_sent_idx[tag['id']] = {'s_idx': [], 'kwe': [], 'kwe_id': [], 'title': ''}

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
                        label_ids_sent_idx[tag['id']]['kwe'].append([span['m']])
                        label_ids_sent_idx[tag['id']]['kwe_id'].append([span['kwe']])
                    else:
                        label_ids_sent_idx[tag['id']]['kwe'][-1].append(span['m'])
                        label_ids_sent_idx[tag['id']]['kwe_id'][-1].append(span['kwe'])
                    label_ids_sent_idx[tag['id']]['title'] = db_label['label_title']
            all_sentences.extend(sentences)
            prev_segment_offset += len(sentences)

    # add a larger (multi-sentence) passage matching keyword expression
    # considering overlaps also ... hence the complicated code
    for passage_size in passage_sizes:
        for label_name, label_data in label_ids_sent_idx.items():
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
                kwe_ids = []
                for j in range(start, end):
                    if j in passage_center_sentence_indices:
                        passage_center_sentence_indices.remove(j)
                        x = label_data['s_idx'].index(j)
                        if x < len(label_data['kwe']):
                            kwes.extend(label_data['kwe'][x])
                        if x < len(label_data['kwe_id']):
                            kwe_ids.extend(label_data['kwe_id'][x])
                    passage.append(all_sentences[j])

                if len(passage) < passage_size:
                    continue
                if len(passage) <= 0:
                    continue

                db_span_label = seed_label.copy()
                db_span_label['kwe_id'] = kwe_ids
                db_span_label['kwe'] = kwes
                db_span_label['label_id'] = label_name
                db_span_label['label_title'] = label_data['title']
                db_span_label['passage_cat'] = passage_size
                db_span_label['passage'] = ' '.join(passage)
                db_labels.append(db_span_label)

    return text, db_labels
