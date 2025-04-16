import logging
import time
from argparse import ArgumentParser
from collections import Counter
from typing import Dict, Any, List, Tuple

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


from ..const import __supported_languages, __label_split_names, __label_splits
from ..embd_model import EmbeddingModelWrapperFactory, EmbeddingModelWrapper
from ..model_data import ModelTestData
from ..utils import compute_arg_collection_name, get_index_path, load_data, init_labeler, filter_metrics, load_labels, \
    filter_samples
from ...core.args import CommonArguments
from ...core.labels import Labeler, MultilabelLabeler
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
        default='tfidf'
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

    metrics = Metrics(f'weak_majority_{args.collection}{suffix}', labeler.get_type_code())

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

    metrics = Metrics(f'weak_random_{args.collection}{suffix}', labeler.get_type_code())

    logger.info(f'Computing metrics')
    y_true_m, y_pred_m = filter_metrics(args, labeler, y_true, y_pred)
    metrics(y_true_m, y_pred_m, 'test/', 0.5)
    meta = {'num_samples': np.shape(y_true_m)[0], 'num_labels': np.shape(y_true_m)[1]}
    metrics.dump(args.data_result_dir, meta, None, 100)
    logger.info(f'Computation done in {(time.time() - t1):8.2f} seconds')

    return 0


def _count_labels(target_labels, data: List[List[str]]):
    instance_counts = []
    label_counts = {}
    for instance_labels in data:
        instance_labels.sort()
        if target_labels:
            instance_labels = [item for item in instance_labels if item in target_labels]
        if not instance_labels:
            continue
        for label in instance_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        instance_counts.append(len(instance_labels))
    return instance_counts, label_counts


def _partition_svm(labeler, train_texts, train_labels, x_test, y_true) -> np.ndarray:
    """
    GPU poor; Split labels into batches of 200ish (column-wise) and merge columns back for complete eval.
    """
    from sklearn.multioutput import MultiOutputClassifier
    from cuml.svm import SVC

    y_pred = []
    # Split labels into batches of 200ish (column-wise)
    #bs = 65  # freq-eurlex
    #bs = 50  # freq-newsmon
    bs = 240
    labels = labeler.encoder.classes_
    assert y_true.shape[0] == x_test.shape[0]
    assert y_true.shape[1] == len(labels)

    for i in range(0, len(labels), bs):
        train_labels_batch = train_labels[:, i:i + bs]
        X_batch = train_texts
        y_batch = train_labels_batch
        # Assert for empty labels in this batch
        zero_in_batch = np.where(np.sum(y_batch, axis=0) == 0)[0]
        if zero_in_batch.size > 0:
            logger.warning(f'Batch {i // bs} has zero columns: {zero_in_batch}')

        t1 = time.time()
        # Train batch classifier
        batch_clf = MultiOutputClassifier(
            SVC(kernel='rbf', C=1.0, gamma='scale', verbose=True),
        )
        batch_clf.fit(X_batch, y_batch)
        logger.warning(f'Train of batch [{y_batch.shape}][{i},{(i + bs)}] done in {(time.time() - t1):8.2f} seconds')

        t1 = time.time()
        y_pred_i = batch_clf.predict(x_test)
        y_pred.append(y_pred_i)
        logger.warning(f'Predicted sample batch [{y_batch.shape}][{i},{(i + bs)}] in {(time.time() - t1):8.2f} seconds')
        del batch_clf

    y_pred = np.concatenate(y_pred, axis=1)
    return y_pred


def bl_svm(args):
    """
    Baseline SVM classifier
    ./newsmon bl svm -c newsmon -l sl --public
    ./newsmon bl svm -c newsmon -l sl --public --test_l_class Rare
    ./newsmon bl svm -c newsmon -l sl --public --test_l_class Frequent
    """
    t0 = time.time()
    compute_arg_collection_name(args)
    train_data_as_dicts, train_df = load_data(args, args.collection + '_train')  # we load the train data
    test_data_as_dicts, test_df = load_data(args, args.collection + '_test')  # we load the test data

    models = EmbeddingModelWrapperFactory.init_models(args)

    suffix = ''
    target_labels = None
    if args.test_l_class != 'all':
        suffix = '_' + args.test_l_class
        label_classes = load_labels(args.data_in_dir, args.collection, __label_splits, __label_split_names)
        target_labels = label_classes[args.test_l_class]

    instance_counts, label_counts = _count_labels(target_labels, train_df['label'].tolist())
    filtered_data, filtered_labels = filter_samples(target_labels, train_data_as_dicts)

    valid_labels = list(label_counts.keys())
    labeler = MultilabelLabeler(valid_labels)
    labeler.fit()
    logger.info(f'Computed labels in {(time.time() - t0):8.2f} seconds')

    train_texts = [x['text'] for x in filtered_data]
    logger.info(f'Computed data {len(train_texts)} samples and {len(labeler.encoder.classes_)}')

    train_labels = labeler.vectorize(filtered_labels)
    zero_label_cols = np.where(np.sum(train_labels, axis=0) == 0)[0]
    logger.info(f'Missing labels {zero_label_cols}')

    test_data, test_labels = filter_samples(valid_labels, test_data_as_dicts)
    test_text = [x['text'] for x in test_data]
    y_true = labeler.vectorize(test_labels)

    for m_name, model in models.items():
        if m_name == 'tfidf':
            model.fit(train_texts)
        t0 = time.time()
        train_texts = model.embed(train_texts)
        logger.info(f'SVM train embeddings {m_name} done in {(time.time() - t0):8.2f} seconds')

        t0 = time.time()
        test_text = model.embed(test_text)
        logger.info(f'SVM test embeddings {m_name} done in {(time.time() - t0):8.2f} seconds')

        t0 = time.time()
        y_pred = _partition_svm(labeler, train_texts, train_labels, test_text, y_true)
        logger.info(f'SVM model {m_name} predict done in {(time.time() - t0):8.2f} seconds')

        t0 = time.time()
        logger.info(f'Computing metrics')
        metrics = Metrics(f'svm_{m_name}_{args.collection}{suffix}', labeler.get_type_code())
        metrics(y_true, y_pred, 'test/', 0.5)
        meta = {'num_samples': np.shape(y_true)[0], 'num_labels': np.shape(y_true)[1]}
        metrics.dump(args.data_result_dir, meta, None, 100)
        logger.info(f'Computation done in {(time.time() - t0):8.2f} seconds')
    return 0


def bl_logreg(args):
    """
    Baseline Logistic Regression classifier
    ./newsmon bl logreg -c newsmon -l sl --public
    ./newsmon bl logreg -c newsmon -l sl --public --test_l_class Rare
    ./newsmon bl logreg -c newsmon -l sl --public --test_l_class Frequent
    """
    t0 = time.time()
    args.tfidf_max_df = 0.8  # from grid search
    compute_arg_collection_name(args)
    train_data_as_dicts, train_df = load_data(args, args.collection + '_train')  # we load the train data
    test_data_as_dicts, test_df = load_data(args, args.collection + '_test')  # we load the test data

    models = EmbeddingModelWrapperFactory.init_models(args)

    suffix = ''
    target_labels = None
    if args.test_l_class != 'all':
        suffix = '_' + args.test_l_class
        label_classes = load_labels(args.data_in_dir, args.collection, __label_splits, __label_split_names)
        target_labels = label_classes[args.test_l_class]

    instance_counts, label_counts = _count_labels(target_labels, train_df['label'].tolist())
    filtered_data, filtered_labels = filter_samples(target_labels, train_data_as_dicts)

    valid_labels = list(label_counts.keys())
    labeler = MultilabelLabeler(valid_labels)
    labeler.fit()
    logger.info(f'Computed labels in {(time.time() - t0):8.2f} seconds')

    train_texts = [x['text'] for x in filtered_data]
    logger.info(f'Computed data {len(train_texts)} samples and {len(labeler.encoder.classes_)}')

    train_labels = labeler.vectorize(filtered_labels)
    zero_label_cols = np.where(np.sum(train_labels, axis=0) == 0)[0]
    logger.info(f'Missing labels {zero_label_cols}')

    test_data, test_labels = filter_samples(valid_labels, test_data_as_dicts)
    test_text = [x['text'] for x in test_data]
    y_true = labeler.vectorize(test_labels)

    from sklearn.multioutput import MultiOutputClassifier
    from cuml.linear_model import LogisticRegression
    for m_name, model in models.items():
        if m_name == 'tfidf':
            model.fit(train_texts)
        t0 = time.time()
        train_texts = model.embed(train_texts)
        logger.info(f'Train embeddings {m_name} done in {(time.time() - t0):8.2f} seconds')

        t0 = time.time()
        test_text = model.embed(test_text)
        logger.info(f'Test embeddings {m_name} done in {(time.time() - t0):8.2f} seconds')

        t0 = time.time()
        clf = MultiOutputClassifier(LogisticRegression(C=1000))
        clf.fit(train_texts, train_labels)
        logger.info(f'LogReg model {m_name} train done in {(time.time() - t0):8.2f} seconds')

        t0 = time.time()
        y_pred = clf.predict(test_text)
        logger.info(f'LogReg model {m_name} predict done in {(time.time() - t0):8.2f} seconds')

        metrics = Metrics(f'logreg_{m_name}_{args.collection}{suffix}', labeler.get_type_code())
        t1 = time.time()
        logger.info(f'LogReg computing {m_name} metrics')
        metrics(y_true, y_pred, 'test/', 0.5)
        meta = {'num_samples': np.shape(y_true)[0], 'num_labels': np.shape(y_true)[1]}
        metrics.dump(args.data_result_dir, meta, None, 100)
        logger.info(f'LogReg computation for {m_name} done in {(time.time() - t1):8.2f} seconds')
    return 0


def bl_logreg_search(args):
    """
    Baseline Logistic Regression TF-IDF classifier grid search.
    ./newsmon bl logreg_search -c newsmon -l sl --public
    ./newsmon bl logreg_search -c newsmon -l sl --public --test_l_class Rare
    ./newsmon bl logreg_search -c newsmon -l sl --public --test_l_class Frequent
    """
    t0 = time.time()
    compute_arg_collection_name(args)
    train_data_as_dicts, train_df = load_data(args, args.collection + '_train')  # we load the train data
    dev_data_as_dicts, dev_df = load_data(args, args.collection + '_dev')  # we load the dev data
    train_data_as_dicts.extend(dev_data_as_dicts)
    test_data_as_dicts, test_df = load_data(args, args.collection + '_test')  # we load the test data

    target_labels = None
    if args.test_l_class != 'all':
        label_classes = load_labels(args.data_in_dir, args.collection, __label_splits, __label_split_names)
        target_labels = label_classes[args.test_l_class]

    instance_counts, label_counts = _count_labels(target_labels, train_df['label'].tolist())
    if target_labels == None:
        target_labels = [k for k, v in label_counts.items() if v > 1]

    filtered_data, filtered_labels = filter_samples(target_labels, train_data_as_dicts)

    labeler = MultilabelLabeler(target_labels)
    labeler.fit()
    logger.info(f'Computed labels in {(time.time() - t0):8.2f} seconds')

    train_texts = [x['text'] for x in filtered_data]
    logger.info(f'Computed data {len(train_texts)} samples and {len(labeler.encoder.classes_)}')

    train_labels = labeler.vectorize(filtered_labels)
    small_label_cols = np.where(np.sum(train_labels, axis=0) < 1)[0]
    logger.info(f'Small num labels {len(small_label_cols)}')

    test_data, test_labels = filter_samples(target_labels, test_data_as_dicts)
    test_text = [x['text'] for x in test_data]
    y_test = labeler.vectorize(test_labels)

    from sklearn.multioutput import MultiOutputClassifier
    from cuml.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import PredefinedSplit
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    #from sklearn.linear_model import LogisticRegression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=2000)))
    ])

    # Define parameter grid for GridSearchCV
    param_grid = {
        #'tfidf__max_df': [0.7, 0.8, 0.9, 1.0],
        'tfidf__max_df': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        #'tfidf__ngram_range': [(1, 1), (1, 2)],
        # 'clf__estimator__C': [0.1, 1, 10]
        #'clf__estimator__C': [10, 100, 1000]
        #'clf__estimator__C': [1000, 10000, 100000]
        #'clf__estimator__C': [2000, 3000, 5000]
        'clf__estimator__C': [0.1, 1, 10, 100, 500, 1000, 2000, 5000, 10000]
    }

    validation_idx = len(train_texts) - len(dev_data_as_dicts)
    train_end_idx = len(train_texts) - 1

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring='f1_micro',  # You can use 'f1_micro', 'f1_macro', or 'accuracy'
        # disable cross validation
        cv=PredefinedSplit(
            np.concatenate(
                [
                    -np.ones(validation_idx, dtype=int),
                    np.zeros(train_end_idx - validation_idx, dtype=int)
                ]
            )
        ),
        verbose=3,
        n_jobs=1
    )

    # Fit grid search
    grid_search.fit(train_texts, train_labels)

    # Predict and evaluate
    y_pred = grid_search.predict(test_text)
    print("Best parameters:", grid_search.best_params_)
    print("Subset Accuracy:", accuracy_score(y_test, y_pred))

    return 0
