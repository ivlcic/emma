import json
import logging
import os
import random
import shutil
import numpy as np
from argparse import ArgumentParser

import pandas as pd
import torch
from lightning import seed_everything
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, ndcg_score
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer, \
    TrainingArguments, Trainer, EvalPrediction

from ..core.metrics import r_precision_at_k
from .ds_utils import _load_data, _compute_output_name
from ..core.args import CommonArguments
from ..core.models import valid_model_names, model_name_map
from ..core.dataset import TruncatedDataset

logger = logging.getLogger('longdoc.hf_train')

corpora = {
    'hyperpartisan', 'eurlex', 'eurlexinv', '20news', 'booksummaries', 'mulabel_sl_p1_s0'
}


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(module_name, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    CommonArguments.train(
        parser, batch=8, epochs=0, lr=3e-5, seed=None
    )
    CommonArguments.num_workers(parser)
    parser.add_argument(
        '--corpus', required=True, help=f'Corpora files to train / validate on {corpora}.'
    )
    parser.add_argument(
        '--model_name', type=str, required=True, help=f'Model name: {valid_model_names}'
    )
    parser.add_argument(
        '--run_id', type=int, help=f'Run id for marking consecutive runs.', default=0
    )
    parser.add_argument(
        '--ckpt', type=str,
        help='Path to a saved ckpt for continued training or evaluation'
             'e.g. bert_hyperpartisan_b8_e20_s3456_lr3e-05--epoch=17.ckpt'
    )


def remove_checkpoint_dir(result_path: str):
    for rd in os.listdir(result_path):
        checkpoint_path = os.path.join(result_path, rd)
        if not rd.startswith('checkpoint'):
            continue
        if not os.path.isdir(checkpoint_path):
            continue
        moved = False
        for f in os.listdir(checkpoint_path):
            source_file_path = os.path.join(checkpoint_path, f)
            if not os.path.isfile(source_file_path):
                continue
            target_file_path = os.path.join(result_path, f)
            shutil.move(source_file_path, target_file_path)
            moved = True
            logger.info('Moved [%s] -> [%s].', source_file_path, target_file_path)
        if moved:
            shutil.rmtree(checkpoint_path)
            logger.info('Removed checkpoint dir [%s].', checkpoint_path)

def main(args) -> int:
    """
    ./longdoc hf_train --batch 8 --epochs 20 --lr 5e-5 --model_name bertmc --corpus eurlex
    ./longdoc hf_train --batch 8 --epochs 20 --lr 5e-5 --model_name xlmrb --corpus eurlex

    ./longdoc hf_train --batch 8 --epochs 20 --lr 5e-5 --model_name tobertmc --corpus eurlex
    ./longdoc hf_train --batch 8 --epochs 20 --lr 5e-5 --model_name toxlmrb --corpus eurlex

    ./longdoc hf_train --batch 8 --epochs 20 --lr 5e-5 --model_name bertmcplusrandom --corpus eurlex
    ./longdoc hf_train --batch 8 --epochs 20 --lr 5e-5 --model_name xlmrbplusrandom --corpus eurlex
    """
    if args.corpus not in corpora:
        raise RuntimeError(f'Corpus {args.corpus} should be one of {corpora}')
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    # torch.set_float32_matmul_precision('high')
    if args.seed is None:
        args.seed = random.randint(1000, 9999)
    logger.debug("Starting training using seed [%s]", args.seed)
    seed_everything(args.seed, workers=True)

    logger.debug("Loading data")

    text_set, label_set, labeler = _load_data(args.data_in_dir, args.corpus)
    logger.debug("Loaded data")
    output_model_name = _compute_output_name(args)
    if args.ckpt:
        output_model_name = args.ckpt.split('--')[0]

    problem_type = 'single_label_classification'
    if 'multilabel' == labeler.get_type_code():
        problem_type = 'multi_label_classification'

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name_map[args.model_name], cache_dir=args.tmp_dir, num_labels=labeler.num_labels,
        id2label=labeler.ids_to_labels(), label2id=labeler.labels_to_ids(),
        problem_type=problem_type
    )
    model.to(device)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name_map[args.model_name], cache_dir=args.tmp_dir
    )

    datasets = {}
    average_labels_per_sample = 0
    for split in ['dev', 'test', 'train']:
        if split not in text_set.keys():
            continue
        datasets[split] = TruncatedDataset(text_set[split], label_set[split], tokenizer, 512)
        average_labels_per_sample += datasets[split].average_labels
    average_labels_per_sample /= 3

    logger.info(f'Loaded train[{len(text_set["train"])}] dev[{len(text_set["dev"])}] test[{len(text_set["test"])}]')
    logger.info(f'Loaded {labeler} with {labeler.num_labels}')

    def preprocess_logits_for_metrics(logits: Tensor, _: Tensor):
        if labeler.get_type_code() == 'multilabel':
            prob = torch.sigmoid(logits)
        else:
            prob = torch.softmax(logits, dim=-1)
        return prob

    log_epochs = []
    avg_k = round(average_labels_per_sample)

    def compute_metrics(eval_pred: EvalPrediction):
        y_true = eval_pred.label_ids
        y_prob = eval_pred.predictions
        if labeler.get_type_code() == 'multilabel':
            y_pred = (y_prob > 0.5).astype(np.float32)
        else:
            y_pred = np.argmax(y_prob, axis=-1)
        logger.info("Epoch: %s", len(log_epochs))
        metric = {}
        for average_type in ['micro', 'macro', 'weighted']:
            if labeler.get_type_code() == 'binary' and not average_type == 'macro':
                continue
            p = precision_score(y_true, y_pred, average=average_type)
            r = recall_score(y_true, y_pred, average=average_type)
            f1 = f1_score(y_true, y_pred, average=average_type)
            metric[f'{average_type}.f1'] = f1
            metric[f'{average_type}.p'] = p
            metric[f'{average_type}.r'] = r
        metric['accuracy'] = accuracy_score(y_true, y_pred)
        metric['m_name'] = output_model_name
        if labeler.get_type_code() == 'multilabel':
            for k in range(1, 10, 2):
                metric[f'r-precision@{k}'], _ = r_precision_at_k(y_true, y_prob, k=k)
                metric[f'ndcg@{k}'] = ndcg_score(y_true, y_prob, k=k)
            metric[f'ndcg'] = ndcg_score(y_true, y_prob)
            metric['hamming_loss'] = hamming_loss(y_true, y_pred)

        log_epochs.append(metric)
        return metric

    result_path = str(os.path.join(args.data_out_dir, output_model_name))
    training_args = TrainingArguments(
        output_dir=result_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        load_best_model_at_end=True,
        save_strategy='epoch',
        learning_rate=args.lr,
        save_total_limit=5,
        #metric_for_best_model='micro.f1',
        #greater_is_better=True,
        logging_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['dev'],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    logger.info(
        f'Will start training {model} using tokenizer {tokenizer} with '
        f'batch size {args.batch}, lr:{args.lr} for {args.epochs} epochs.'
    )
    trainer.train()
    pd.DataFrame(log_epochs).to_csv(os.path.join(result_path, 'train_metrics.csv'))

    predictions = trainer.predict(datasets['test'])
    metrics = predictions.metrics
    log_epochs.append(metrics)
    pd.DataFrame(log_epochs).to_csv(os.path.join(result_path, output_model_name + '_metrics.csv'))
    with open(os.path.join(result_path, output_model_name + '_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {'epochs': log_epochs, 'seed': args.seed, 'model_name': output_model_name},
            f, ensure_ascii=False, indent=2, sort_keys=False,
        )

    # remove_checkpoint_dir(result_path)
    return 0
