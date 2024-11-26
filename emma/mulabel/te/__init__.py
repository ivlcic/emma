import os
import logging
from typing import Any, Tuple

import torch
import random

from argparse import ArgumentParser

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer, \
    TrainingArguments, Trainer, EvalPrediction
from lightning import seed_everything

from ...core.args import CommonArguments
from ...core.models import valid_model_names, model_name_map
from ...core.metrics import Metrics
from ...core.wandb import initialize_run
from ..utils import __supported_languages, compute_arg_collection_name, compute_model_output_name, load_train_data, \
    construct_datasets

logger = logging.getLogger('mulabel.te_train')


# noinspection DuplicatedCode
def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(module_name, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
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
        '--passage_size', help='When calibrating use passage_size',
        type=int, default=1, choices=[1, 3, 5, 7, 9, 0]
    )
    CommonArguments.train(parser, batch=8, epochs=0, lr=3e-5, seed=None)
    CommonArguments.num_workers(parser)
    parser.add_argument(
        '--ptm_name', type=str, required=True, help=f'Pretrained model name: {valid_model_names}'
    )
    parser.add_argument(
        '--run_id', type=int, help=f'Run id for marking consecutive runs.', default=0
    )
    parser.add_argument(
        '--ckpt', type=str,
        help='Path to a saved ckpt for continued training or evaluation'
             'e.g. bert_hyperpartisan_b8_e20_s3456_lr3e-05--epoch=17.ckpt'
    )


def init_task(args) -> Tuple[str, Any]:
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir
    os.environ['WANDB_LOG_MODEL'] = 'end'
    os.environ['WANDB_WATCH'] = 'false'

    compute_arg_collection_name(args)  # combine args.params to full collection name
    args.corpus = args.collection  # use full collection name as corpus name

    output_model_name = compute_model_output_name(args)  # append training args
    if args.ckpt:
        output_model_name = args.ckpt.split('--')[0]

    tags = [
        args.ptm_name, model_name_map[args.ptm_name], args.collection_conf, args.corpus,
        f's{args.seed}', f'e{args.epochs}', f'e{args.batch}', f'lr{args.lr}'
    ]
    if args.public:
        tags.append('public')
    if args.seed_only:
        tags.append('seed_labels')

    params = {
        'job_type': 'encoder_train',
        'name': output_model_name,
        'run_id': output_model_name + '@' + str(args.run_id),
        'run_group': args.collection_conf,
        'tags': tags,
        'conf': {
            'ptm_alias': args.ptm_name,
            'ptm': model_name_map[args.ptm_name],
            'lang': args.lang_conf,
            'corpus': args.corpus,
            'seed': args.seed,
            'epochs': args.epochs,
            'batch': args.batch,
            'learning_rate': args.lr
        }
    }

    return output_model_name, initialize_run(**params)


# noinspection DuplicatedCode
def te_train(args) -> int:
    """
    Train transformer encoder models
    ./mulabel te train --batch 16 --epochs 30 --lr 3e-5 \
       --ptm_name xlmrb -c eurlex --num_workers 4 --seed 7681 --run_id 1
    ./mulabel te train --batch 16 --epochs 30 --lr 3e-5 \
       --ptm_name xlmrb -c mulabel -l sl --public --num_workers 4 --seed 4823 --run_id 1
    """

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    # torch.set_float32_matmul_precision('high')
    if args.seed is None:
        args.seed = random.randint(1000, 9999)
    logger.debug('Starting training using seed [%s]', args.seed)
    seed_everything(args.seed, workers=True)

    output_model_name, run = init_task(args)

    logger.debug(f'Loading data from corpus [{args.corpus}]')
    text_set, label_set, labeler = load_train_data(args.data_in_dir, args.corpus)
    logger.debug(f'Loaded data from corpus [{args.corpus}]')

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name_map[args.ptm_name], cache_dir=args.tmp_dir, num_labels=labeler.num_labels,
        id2label=labeler.ids_to_labels(), label2id=labeler.labels_to_ids(),
        problem_type='multi_label_classification' if 'multilabel' == labeler.get_type_code()
        else 'single_label_classification'
    )
    model.to(device)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name_map[args.ptm_name], cache_dir=args.tmp_dir
    )

    datasets, avg_k = construct_datasets(text_set, label_set, tokenizer, 512)
    logger.info(f'Loaded train[{len(datasets["train"])}] dev[{len(datasets["dev"])}] test[{len(datasets["test"])}]')
    logger.info(f'Loaded {labeler} with {labeler.num_labels}')

    def preprocess_logits_for_metrics(logits: Tensor, _: Tensor):
        if labeler.get_type_code() == 'multilabel':
            prob = torch.sigmoid(logits)
        else:
            prob = torch.softmax(logits, dim=-1)
        return prob

    metrics = Metrics(output_model_name, labeler.get_type_code(), avg_k)

    def compute_metrics(eval_pred: EvalPrediction):
        y_true = eval_pred.label_ids
        y_prob = eval_pred.predictions
        return metrics(y_true, y_prob)

    result_path = str(os.path.join(args.data_out_dir, output_model_name))
    training_args = TrainingArguments(
        output_dir=result_path,
        report_to='wandb',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        eval_strategy="epoch",
        disable_tqdm=False,
        load_best_model_at_end=True,
        save_strategy='epoch',
        learning_rate=args.lr,
        save_total_limit=3,
        logging_strategy='epoch',
        logging_steps=1,
        run_name=output_model_name,
        metric_for_best_model='micro.f1'
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
    logger.info(
        f'Training done for {model} using tokenizer {tokenizer} with '
        f'batch size {args.batch}, lr:{args.lr} for {args.epochs} epochs.'
    )
    trainer.predict(datasets['test'])
    metrics.dump(result_path, {'seed': args.seed}, run)
    # remove_checkpoint_dir(result_path)
    return 0


def te_test(args) -> int:
    """
    ./mulabel te test --ptm_name xlmrb_news_sl-p1s0_x0_b16_e30_s1425_lr3e-05 -c mulabel_sl_p1_s0_filtered_article
    ./mulabel te test --ptm_name xlmrb_news_sl-p1s0_x0_b16_e30_s1710_lr3e-05 -c mulabel_sl_p1_s0_filtered_article
    ./mulabel te test --ptm_name xlmrb_news_sl-p1s0_x0_b16_e30_s3821_lr3e-05 -c mulabel_sl_p1_s0_filtered_article
    ./mulabel te test --ptm_name xlmrb_news_sl-p1s0_x0_b16_e30_s4823_lr3e-05 -c mulabel_sl_p1_s0_filtered_article
    ./mulabel te test --ptm_name xlmrb_news_sl-p1s0_x0_b16_e30_s5327_lr3e-05 -c mulabel_sl_p1_s0_filtered_article

    ./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s1710_lr3e-05 -c mulabel_sl_p1_s0
    ./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s2573_lr3e-05 -c mulabel_sl_p1_s0
    ./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s3821_lr3e-05 -c mulabel_sl_p1_s0
    ./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s4823_lr3e-05 -c mulabel_sl_p1_s0
    ./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s7352_lr3e-05 -c mulabel_sl_p1_s0

    ./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2611_lr3e-05 -c eurlex_all_p0_s0
    ./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2963_lr3e-05 -c eurlex_all_p0_s0
    ./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s4789_lr3e-05 -c eurlex_all_p0_s0
    ./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s5823_lr3e-05 -c eurlex_all_p0_s0
    ./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s7681_lr3e-05 -c eurlex_all_p0_s0

    """
    model_path = os.path.join(args.data_out_dir, 'test', args.ptm_name)
    if not os.path.exists(model_path):
        raise ValueError(f'Missing model path: {model_path}')
    args.corpus = args.collection
    logger.debug(f'Loading data from corpus [{args.corpus}]')
    text_set, label_set, labeler = load_train_data(args.data_in_dir, args.corpus, test_only=True)
    logger.debug(f'Loaded data from corpus [{args.corpus}]')

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_path, cache_dir=args.tmp_dir, num_labels=labeler.num_labels,
        id2label=labeler.ids_to_labels(), label2id=labeler.labels_to_ids(),
        problem_type='multi_label_classification' if 'multilabel' == labeler.get_type_code()
        else 'single_label_classification'
    )
    model.to(device)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        'xlm-roberta-base', cache_dir=args.tmp_dir
    )

    datasets, avg_k = construct_datasets(text_set, label_set, tokenizer, 512)

    def preprocess_logits_for_metrics(logits: Tensor, _: Tensor):
        if labeler.get_type_code() == 'multilabel':
            prob = torch.sigmoid(logits)
        else:
            prob = torch.softmax(logits, dim=-1)
        return prob

    metrics = Metrics(args.ptm_name, labeler.get_type_code(), avg_k)

    def compute_metrics(eval_pred: EvalPrediction):
        y_true = eval_pred.label_ids
        y_prob = eval_pred.predictions
        return metrics(y_true, y_prob)

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.predict(datasets['test'])
    #m = metrics(np.array(y_true, dtype=float), np.array(y_pred, dtype=float), 'test/')
    metrics.dump(args.data_out_dir, None, None)
    return 0
