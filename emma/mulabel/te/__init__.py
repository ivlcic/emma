import os
import logging
import torch
import random

from argparse import ArgumentParser

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer, \
    TrainingArguments, Trainer, EvalPrediction
from lightning import seed_everything

from ...core.dataset import TruncatedDataset
from ...core.metrics import Metrics
from ...core.args import CommonArguments
from ...core.models import valid_model_names, model_name_map
from ..utils import __supported_languages, compute_arg_collection_name
from .utils import _load_data, _compute_output_name

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


def init_task(args) -> str:
    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir
    os.environ['WANDB_LOG_MODEL'] = 'end'
    os.environ['WANDB_WATCH'] = 'false'

    compute_arg_collection_name(args)  # combine args.params to full collection name
    args.corpus = args.collection  # use full collection name as corpus name

    output_model_name = _compute_output_name(args)  # append training args
    if args.ckpt:
        output_model_name = args.ckpt.split('--')[0]

    tags = [
        args.lang, args.ptm_name, args.corpus, f's{args.seed}', f'e{args.epochs}', f'e{args.batch}', f'lr{args.lr}'
    ]
    if args.public:
        tags.append('public')
    if args.seed_only:
        tags.append('seed_labels')

    api_key = os.getenv('WANDB_API_KEY')
    if api_key is not None:
        import wandb
        wandb.login('never', api_key)
        wandb.init(
            'encoder_train',
            project=os.getenv('WANDB_PROJECT'),
            name=output_model_name,
            id=output_model_name + '@' + str(args.run_id),
            group=args.collection_conf,
            tags=tags,
            config={
                'ptm_alias': args.ptm_name,
                'lang': args.lang_conf,
                'corpus': args.corpus,
                'seed': args.seed,
                'epochs': args.epochs,
                'batch': args.batch,
                'learning_rate': args.lr
            }
        )

    return output_model_name


# noinspection DuplicatedCode
def te_train(args) -> int:
    """
    Train transformer encoder models
    ./mulabel te train --batch 8 --epochs 20 --lr 5e-5 --run_id 1 --ptm_name bertmc -c eurlex
    ./mulabel te train --batch 8 --epochs 20 --lr 5e-5 --run_id 1 --ptm_name xlmrb -c eurlex
    """

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    # torch.set_float32_matmul_precision('high')
    if args.seed is None:
        args.seed = random.randint(1000, 9999)
    logger.debug('Starting training using seed [%s]', args.seed)
    seed_everything(args.seed, workers=True)

    output_model_name = init_task(args)

    logger.debug(f'Loading data from corpus [{args.corpus}]')
    text_set, label_set, labeler = _load_data(args.data_in_dir, args.corpus)
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

    datasets = {}
    average_labels_per_sample = 0
    for split in ['dev', 'test', 'train']:
        if split not in text_set.keys():
            continue
        datasets[split] = TruncatedDataset(text_set[split], label_set[split], tokenizer, 512)
        average_labels_per_sample += datasets[split].average_labels
    average_labels_per_sample /= 3
    avg_k = round(average_labels_per_sample)

    logger.info(f'Loaded train[{len(text_set["train"])}] dev[{len(text_set["dev"])}] test[{len(text_set["test"])}]')
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
        evaluation_strategy="epoch",
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
        train_dataset=datasets['test'],
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
    metrics.dump(result_path, {'seed': args.seed})
    # remove_checkpoint_dir(result_path)
    return 0
