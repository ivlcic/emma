import logging
import time
from argparse import ArgumentParser
from collections import Counter
from typing import Dict

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


from ..const import __supported_languages, __label_split_names, __label_splits
from ..embd_model import EmbeddingModelWrapperFactory, EmbeddingModelWrapper
from ..model_data import ModelTestData
from ..utils import compute_arg_collection_name, get_index_path, load_data, init_labeler, filter_metrics, load_labels
from ...core.args import CommonArguments
from ...core.labels import Labeler
from ...core.metrics import Metrics
from ...core.models import retrieve_model_name_map

logger = logging.getLogger('newsmon.bl')


# noinspection DuplicatedCode
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
        '--test_l_class', type=str, help=f'Test specified label class.',
        choices=['all'].extend(__label_split_names), default='all'
    )
    parser.add_argument(
        '--ptm_models',
        help=f'Use only ptm_models (filter everything else out). '
             f'You can use a comma separated list of {retrieve_model_name_map.keys()}',
        type=str,
        default='bge_m3'
    )


# noinspection DuplicatedCode
def init_model_data(args,
                    labeler: Labeler,
                    index_metric: int,
                    prefix: str,
                    models: Dict[str, EmbeddingModelWrapper]) -> Dict[str, ModelTestData]:
    index_path = get_index_path(args)
    # static model data
    model_data = {}
    for m_name, model in models.items():
        t0 = time.time()
        suffix = ''
        if args.test_l_class != 'all':
            suffix = '_' + args.test_l_class
        m = Metrics(f'{prefix}_{m_name}_{args.collection}{suffix}', labeler.get_type_code())
        model_data[m_name] = ModelTestData(m_name, str(index_path), model, index_metric, m)
        logger.info(f'Loaded {m_name} data in {(time.time() - t0):8.2f} seconds')
    return model_data


# noinspection DuplicatedCode
def bl_test_zshot(args) -> int:
    """
    ./newsmon bl test_zshot -c newsmon -l sl --public --ptm_models bge_m3,jinav3,gte
    """
    compute_arg_collection_name(args)
    models = EmbeddingModelWrapperFactory.init_models(args)
    labeler, _ = init_labeler(args)

    model_data = init_model_data(args, labeler, faiss.METRIC_L2, 'zshot', models)

    batch_size = 384
    t0 = time.time()
    for model_name, m_data in model_data.items():
        logger.info(f'Processing zero-shot eval for {model_name}')
        for start_idx in tqdm(range(0, m_data.test_data['count'], batch_size), desc='Processing zero-shot eval.'):
            end_idx = min(start_idx + batch_size, m_data.test_data['count'])
            query_vectors = m_data.test_data['x'][start_idx:end_idx]
            yl_true = m_data.test_data['y_true'][start_idx:end_idx]

            # Search for the topk nearest neighbors for all query vectors in the batch
            # noinspection PyArgumentList
            sim, indices = m_data.index.search(query_vectors, 1)  # Batched search
            yl_pred = m_data.train_data['y_true'][indices].squeeze()
            m_data.y_pred.extend(yl_pred)
            m_data.y_true.extend(yl_true)

    logger.info(f'Measured performance in {(time.time() - t0):8.2f} seconds')
    logger.info(f'Computing metrics')
    for model_name, m_data in model_data.items():
        logger.info(f'Processing zero-shot metrics for {model_name}')
        y_true_m, y_pred_m = filter_metrics(args, labeler, m_data.y_true, m_data.y_pred)
        m_data.metrics(y_true_m, y_pred_m, 'test/')
        meta = {'num_samples': np.shape(y_true_m)[0], 'num_labels': np.shape(y_true_m)[1]}
        m_data.metrics.dump(args.data_result_dir, meta, None, 100)

    logger.info(f'Computation done in {(time.time() - t0):8.2f} seconds')
    return 0


def bl_majority(args):
    """
    Majority classifier
    ./newsmon bl majority -c newsmon -l sl --public
    ./newsmon bl majority -c newsmon -l sl --public --test_l_class Rare
    ./newsmon bl majority -c newsmon -l sl --public --test_l_class Frequent
    """
    compute_arg_collection_name(args)
    labeler, labels_df = init_labeler(args)
    train_data_as_dicts, train_df = load_data(args, args.collection + '_train')  # we load the train data
    dev_data_as_dicts, dev_df = load_data(args, args.collection + '_dev')  # we load the validation data
    test_data_as_dicts, test_df = load_data(args, args.collection + '_test')  # we load the test data

    data_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    suffix = ''
    target_labels = None
    if args.test_l_class != 'all':
        suffix = '_' + args.test_l_class
        label_classes = load_labels(args.data_in_dir, args.collection, __label_splits, __label_split_names)
        target_labels = label_classes[args.test_l_class]

    instance_counts = []
    for instance_labels in data_df['label']:
        instance_labels.sort()
        if target_labels:
            instance_labels = [item for item in instance_labels if item in target_labels]
        if not instance_labels:
            continue
        instance_counts.append(len(instance_labels))

    std_dev_labels = np.std(instance_counts, axis=0)
    mean_labels = np.mean(instance_counts, axis=0)
    logger.info(f'Mean {mean_labels} Standard deviation {std_dev_labels}')

    if target_labels:
        labels_df = labels_df[labels_df['label'].isin(target_labels)]

    top_labels = labels_df.nlargest(round(mean_labels), 'count')
    predicted_labels = top_labels['label'].tolist()
    logger.info(f'Top labels: {top_labels}, {predicted_labels}, {type(predicted_labels)}')
    y_true = []
    y_pred = []
    t1 = time.time()
    for data in test_data_as_dicts:
        true_labels = data['label']
        if target_labels:
            true_labels = [item for item in true_labels if item in target_labels]
        if not true_labels:
            continue
        y_true_i = labeler.vectorize([true_labels])
        y_pred_i = labeler.vectorize([predicted_labels])
        y_true.append(y_true_i)
        y_pred.append(y_pred_i)

    metrics = Metrics(f'bl_majority_{args.collection}{suffix}', labeler.get_type_code())

    logger.info(f'Computing metrics')
    y_true_m, y_pred_m = filter_metrics(args, labeler, y_true, y_pred)
    metrics(y_true_m, y_pred_m, 'test/', 0.5)
    meta = {'num_samples': np.shape(y_true_m)[0], 'num_labels': np.shape(y_true_m)[1]}
    metrics.dump(args.data_result_dir, meta, None, 100)
    logger.info(f'Computation done in {(time.time() - t1):8.2f} seconds')

    return 0


def bl_random(args):
    """
    Random classifier
    ./newsmon bl random -c newsmon -l sl --public
    ./newsmon bl random -c newsmon -l sl --public --test_l_class Rare
    ./newsmon bl random -c newsmon -l sl --public --test_l_class Frequent
    """
    compute_arg_collection_name(args)
    labeler, labels_df = init_labeler(args)
    train_data_as_dicts, train_df = load_data(args, args.collection + '_train')  # we load the train data
    dev_data_as_dicts, dev_df = load_data(args, args.collection + '_dev')  # we load the validation data
    test_data_as_dicts, test_df = load_data(args, args.collection + '_test')  # we load the test data

    data_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    suffix = ''
    target_labels = None
    if args.test_l_class != 'all':
        suffix = '_' + args.test_l_class
        label_classes = load_labels(args.data_in_dir, args.collection, __label_splits, __label_split_names)
        target_labels = label_classes[args.test_l_class]

    combinations = []
    sample_tag_counts = []
    for instance_labels in data_df['label']:
        instance_labels.sort()
        if target_labels:
            instance_labels = [item for item in instance_labels if item in target_labels]
        if not instance_labels:
            continue
        sample_tag_counts.append(len(instance_labels))
        combinations.append(tuple(instance_labels))

    std_dev_labels = np.std(sample_tag_counts, axis=0)
    mean_labels = np.mean(sample_tag_counts, axis=0)

    combo_counts = Counter(combinations)
    total = len(combinations)

    label_combos = list(combo_counts.keys())
    probabilities = np.array([count / total for count in combo_counts.values()])

    logger.info(f'Mean {mean_labels} Standard deviation {std_dev_labels}, combinations {total}')
    y_true = []
    y_pred = []
    t1 = time.time()
    for data in test_data_as_dicts:
        true_labels = data['label']
        if target_labels:
            true_labels = [item for item in true_labels if item in target_labels]
        if not true_labels:
            continue
        combo_idx = np.random.choice(
            np.arange(0, len(label_combos)),
            size=1,
            p=probabilities
        )
        predicted_labels = label_combos[combo_idx[0]]
        predicted_labels = list(predicted_labels)
        y_true_i = labeler.vectorize([true_labels])
        y_pred_i = labeler.vectorize([predicted_labels])
        y_true.append(y_true_i)
        y_pred.append(y_pred_i)

    metrics = Metrics(f'bl_random_{args.collection}{suffix}', labeler.get_type_code())

    logger.info(f'Computing metrics')
    y_true_m, y_pred_m = filter_metrics(args, labeler, y_true, y_pred)
    metrics(y_true_m, y_pred_m, 'test/', 0.5)
    meta = {'num_samples': np.shape(y_true_m)[0], 'num_labels': np.shape(y_true_m)[1]}
    metrics.dump(args.data_result_dir, meta, None, 100)
    logger.info(f'Computation done in {(time.time() - t1):8.2f} seconds')

    return 0


def bl_svm(args):
    """
    Baseline SVM classifier
    ./newsmon bl svm -c newsmon -l sl --public
    ./newsmon bl svm -c newsmon -l sl --public --test_l_class Rare
    ./newsmon bl svm -c newsmon -l sl --public --test_l_class Frequent
    """
    t0 = time.time()
    compute_arg_collection_name(args)
    labeler, labels_df = init_labeler(args)
    train_data_as_dicts, train_df = load_data(args, args.collection + '_train')  # we load the train data
    dev_data_as_dicts, dev_df = load_data(args, args.collection + '_dev')  # we load the validation data
    test_data_as_dicts, test_df = load_data(args, args.collection + '_test')  # we load the test data

    data_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    suffix = ''
    target_labels = None
    if args.test_l_class != 'all':
        suffix = '_' + args.test_l_class
        label_classes = load_labels(args.data_in_dir, args.collection, __label_splits, __label_split_names)
        target_labels = label_classes[args.test_l_class]

    instance_counts = []
    for instance_labels in data_df['label']:
        instance_labels.sort()
        if target_labels:
            instance_labels = [item for item in instance_labels if item in target_labels]
        if not instance_labels:
            continue
        instance_counts.append(len(instance_labels))

    std_dev_labels = np.std(instance_counts, axis=0)
    mean_labels = np.mean(instance_counts, axis=0)
    logger.info(f'Mean {mean_labels} Standard deviation {std_dev_labels}')

    if target_labels:
        labels_df = labels_df[labels_df['label'].isin(target_labels)]

    top_labels = labels_df.nlargest(round(mean_labels), 'count')
    predicted_labels = top_labels['label'].tolist()
    logger.info(f'Top labels: {top_labels}, {predicted_labels}, {type(predicted_labels)}')

    train_texts = [x['text'] for x in train_data_as_dicts]
    #train_texts.extend([x['text'] for x in dev_data_as_dicts])
    train_labels = [x['label'] for x in train_data_as_dicts]
    #train_labels.extend([x['label'] for x in dev_data_as_dicts])

    import cupy as cp
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from cuml.svm import SVC

    tfidf = TfidfVectorizer(analyzer='word', max_features=10000)

    train_texts = tfidf.fit_transform(train_texts)
    train_labels = labeler.vectorize(train_labels)

    # train_texts = train_texts.todense()
    # train_texts = cp.array(train_texts.todense())
    #train_labels = cp.array(train_labels)

    gpu_svm = SVC(
        #kernel='linear',
        #C=1.0,
        probability=False,
        output_type='numpy',
        verbose=True
    )

    logger.info(f'SVM train start in {(time.time() - t0):8.2f} seconds')
    t0 = time.time()
    clf = MultiOutputClassifier(gpu_svm)
    clf.fit(train_texts, train_labels)
    logger.info(f'SVM train done in {(time.time() - t0):8.2f} seconds')
    y_true = []
    y_pred = []
    t1 = time.time()
    for data in test_data_as_dicts:
        true_labels = data['label']
        if target_labels:
            true_labels = [item for item in true_labels if item in target_labels]
        if not true_labels:
            continue
        test_text = tfidf.transform(data['text'])
        y_true_i = labeler.vectorize(true_labels)
        logger.info(f'Dim true {y_true_i.shape}')
        y_true.append(y_true_i)
        test_text = test_text.todense()
        #test_text = cp.array(test_text)
        y_pred_i = clf.predict(test_text)
        #y_pred_i = cp.asnumpy(y_pred_i)
        logger.info(f'Dim pred {y_pred_i.shape}')
        y_pred.append(y_pred_i)

    suffix = ''
    if args.test_l_class != 'all':
        suffix = '_' + args.test_l_class
    metrics = Metrics(f'bl_svm_{args.collection}{suffix}', labeler.get_type_code())

    logger.info(f'Computing metrics')
    y_true_m, y_pred_m = filter_metrics(args, labeler, y_true, y_pred)
    metrics(y_true_m, y_pred_m, 'test/', 0.5)
    meta = {'num_samples': np.shape(y_true_m)[0], 'num_labels': np.shape(y_true_m)[1]}
    metrics.dump(args.data_result_dir, meta, None, 100)
    logger.info(f'Computation done in {(time.time() - t1):8.2f} seconds')
    return 0


def bl_svm2(args):
    """
    Baseline SVM classifier
    ./newsmon bl svm2 -c newsmon -l sl --public
    ./newsmon bl svm2 -c newsmon -l sl --public --test_l_class Rare
    ./newsmon bl svm2 -c newsmon -l sl --public --test_l_class Frequent
    """
    t0 = time.time()
    compute_arg_collection_name(args)
    labeler, labels_df = init_labeler(args)
    train_data_as_dicts, train_df = load_data(args, args.collection + '_train')  # we load the train data
    dev_data_as_dicts, dev_df = load_data(args, args.collection + '_dev')  # we load the validation data
    test_data_as_dicts, test_df = load_data(args, args.collection + '_test')  # we load the test data

    data_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    suffix = ''
    target_labels = None
    if args.test_l_class != 'all':
        suffix = '_' + args.test_l_class
        label_classes = load_labels(args.data_in_dir, args.collection, __label_splits, __label_split_names)
        target_labels = label_classes[args.test_l_class]

    instance_counts = []
    for instance_labels in data_df['label']:
        instance_labels.sort()
        if target_labels:
            instance_labels = [item for item in instance_labels if item in target_labels]
        if not instance_labels:
            continue
        instance_counts.append(len(instance_labels))

    std_dev_labels = np.std(instance_counts, axis=0)
    mean_labels = np.mean(instance_counts, axis=0)
    logger.info(f'Mean {mean_labels} Standard deviation {std_dev_labels}')

    if target_labels:
        labels_df = labels_df[labels_df['label'].isin(target_labels)]

    top_labels = labels_df.nlargest(round(mean_labels), 'count')
    predicted_labels = top_labels['label'].tolist()
    logger.info(f'Top labels: {top_labels}, {predicted_labels}, {type(predicted_labels)}')

    train_texts = [x['text'] for x in train_data_as_dicts]
    #train_texts.extend([x['text'] for x in dev_data_as_dicts])
    train_labels = [x['label'] for x in train_data_as_dicts]
    #train_labels.extend([x['label'] for x in dev_data_as_dicts])

    import cupy as cp
    #from cuml.feature_extraction.text import TfidfVectorizer
    from sklearn.multioutput import MultiOutputClassifier
    from cuml.svm import SVC

    tfidf = TfidfVectorizer(max_features=10000)

    train_texts = tfidf.fit_transform(train_texts)
    train_labels = labeler.vectorize(train_labels)

    # Move data to GPU
    train_texts = cp.sparse.csr_matrix(train_texts).astype(cp.float32)
    train_labels = cp.asarray(train_labels).astype(cp.int32)

    # 3. Filter empty labels
    # Create mask for samples with at least one label
    non_empty_mask = cp.any(train_labels, axis=1)
    non_empty_indices = cp.where(non_empty_mask)[0]  # Get indices of non-empty samples

    # 4. Apply filtering
    train_texts = train_texts[non_empty_indices]  # Works with cuML sparse matrices
    train_labels = train_labels[non_empty_indices]

    svc = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        verbose=1
    )

    logger.info(f'SVM train start in {(time.time() - t0):8.2f} seconds')
    t0 = time.time()
    clf = MultiOutputClassifier(svc)
    clf.fit(train_texts, train_labels)
    logger.info(f'SVM train done in {(time.time() - t0):8.2f} seconds')
    y_true = []
    y_pred = []
    t1 = time.time()
    for data in test_data_as_dicts:
        true_labels = data['label']
        if target_labels:
            true_labels = [item for item in true_labels if item in target_labels]
        if not true_labels:
            continue
        test_text = tfidf.transform(data['text'])
        y_true_i = labeler.vectorize(true_labels)
        logger.info(f'Dim true {y_true_i.shape}')
        y_true.append(y_true_i)
        test_text = cp.sparse.csr_matrix(test_text).astype(cp.float32)
        #test_text = cp.array(test_text)
        y_pred_i = clf.predict(test_text)
        y_pred_i = cp.asnumpy(y_pred_i)
        logger.info(f'Dim pred {y_pred_i.shape}')
        y_pred.append(y_pred_i)

    suffix = ''
    if args.test_l_class != 'all':
        suffix = '_' + args.test_l_class
    metrics = Metrics(f'bl_svm_{args.collection}{suffix}', labeler.get_type_code())

    logger.info(f'Computing metrics')
    y_true_m, y_pred_m = filter_metrics(args, labeler, y_true, y_pred)
    metrics(y_true_m, y_pred_m, 'test/', 0.5)
    meta = {'num_samples': np.shape(y_true_m)[0], 'num_labels': np.shape(y_true_m)[1]}
    metrics.dump(args.data_result_dir, meta, None, 100)
    logger.info(f'Computation done in {(time.time() - t1):8.2f} seconds')
    return 0



def bl_logreg(args):
    """
    Baseline Logistic Regression classifier
    ./newsmon bl logreg -c newsmon -l sl --public
    ./newsmon bl logreg -c newsmon -l sl --public --test_l_class Rare
    ./newsmon bl logreg -c newsmon -l sl --public --test_l_class Frequent
    """
    t0 = time.time()
    compute_arg_collection_name(args)
    labeler, labels_df = init_labeler(args)
    train_data_as_dicts, train_df = load_data(args, args.collection + '_train')  # we load the train data
    dev_data_as_dicts, dev_df = load_data(args, args.collection + '_dev')  # we load the validation data
    test_data_as_dicts, test_df = load_data(args, args.collection + '_test')  # we load the test data

    data_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    suffix = ''
    target_labels = None
    if args.test_l_class != 'all':
        suffix = '_' + args.test_l_class
        label_classes = load_labels(args.data_in_dir, args.collection, __label_splits, __label_split_names)
        target_labels = label_classes[args.test_l_class]

    instance_counts = []
    for instance_labels in data_df['label']:
        instance_labels.sort()
        if target_labels:
            instance_labels = [item for item in instance_labels if item in target_labels]
        if not instance_labels:
            continue
        instance_counts.append(len(instance_labels))

    std_dev_labels = np.std(instance_counts, axis=0)
    mean_labels = np.mean(instance_counts, axis=0)
    logger.info(f'Mean {mean_labels} Standard deviation {std_dev_labels}')

    if target_labels:
        labels_df = labels_df[labels_df['label'].isin(target_labels)]

    top_labels = labels_df.nlargest(round(mean_labels), 'count')
    predicted_labels = top_labels['label'].tolist()
    logger.info(f'Top labels: {top_labels}, {predicted_labels}, {type(predicted_labels)}')

    train_texts = [x['text'] for x in train_data_as_dicts]
    #train_texts.extend([x['text'] for x in dev_data_as_dicts])
    train_labels = [x['label'] for x in train_data_as_dicts]
    #train_labels.extend([x['label'] for x in dev_data_as_dicts])

    import cupy as cp
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml.linear_model import LogisticRegression

    tfidf = TfidfVectorizer(analyzer='word', max_features=10000)

    train_texts = tfidf.fit_transform(train_texts)
    train_labels = labeler.vectorize(train_labels)

    logger.info(f'SVM train start in {(time.time() - t0):8.2f} seconds')
    t0 = time.time()
    clf = LogisticRegression()
    clf.fit(train_texts, cp.asarray(train_labels))
    logger.info(f'SVM train done in {(time.time() - t0):8.2f} seconds')
    y_true = []
    y_pred = []
    t1 = time.time()
    for data in test_data_as_dicts:
        true_labels = data['label']
        if target_labels:
            true_labels = [item for item in true_labels if item in target_labels]
        if not true_labels:
            continue
        test_text = tfidf.transform(data['text'])
        y_true_i = labeler.vectorize(true_labels)
        logger.info(f'Dim true {y_true_i.shape}')
        y_true.append(y_true_i)
        #test_text = test_text.todense()
        #test_text = cp.array(test_text)
        y_pred_i = clf.predict(test_text)
        y_pred_i = cp.asnumpy(y_pred_i)
        logger.info(f'Dim pred {y_pred_i.shape}')
        y_pred.append(y_pred_i)

    suffix = ''
    if args.test_l_class != 'all':
        suffix = '_' + args.test_l_class
    metrics = Metrics(f'bl_logreg_{args.collection}{suffix}', labeler.get_type_code())

    logger.info(f'Computing metrics')
    y_true_m, y_pred_m = filter_metrics(args, labeler, y_true, y_pred)
    metrics(y_true_m, y_pred_m, 'test/', 0.5)
    meta = {'num_samples': np.shape(y_true_m)[0], 'num_labels': np.shape(y_true_m)[1]}
    metrics.dump(args.data_result_dir, meta, None, 100)
    logger.info(f'Computation done in {(time.time() - t1):8.2f} seconds')
    return 0
