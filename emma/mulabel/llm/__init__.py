import os
import logging
from typing import Any, Tuple, Union

import torch
import random

from argparse import ArgumentParser
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from torch import Tensor, optim
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, \
    PreTrainedModel, EvalPrediction, TrainingArguments, Trainer
from lightning import seed_everything

from ...core.args import CommonArguments
from ...core.models import llm_model_name_map
from ...core.metrics import Metrics
from ...core.wandb import initialize_run
from ..utils import __supported_languages, compute_arg_collection_name, compute_model_output_name, load_train_data, \
    construct_datasets

logger = logging.getLogger('mulabel.te_train')

__peft_confs = {
    'llama3b': LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias='none'
    )
}


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
        '--ptm_name', type=str, required=True, help=f'Pretrained model name: {llm_model_name_map.keys()}'
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
    hf_token = os.getenv('HF_TOKEN')
    if hf_token is not None:
        from huggingface_hub import login
        login(hf_token)

    os.environ['HF_HOME'] = args.tmp_dir  # local tmp dir
    os.environ['WANDB_LOG_MODEL'] = 'end'
    os.environ['WANDB_WATCH'] = 'false'

    compute_arg_collection_name(args)  # combine args.params to full collection name
    args.corpus = args.collection  # use full collection name as corpus name
    args.ptm_model_name = llm_model_name_map[args.ptm_name]
    output_model_name = compute_model_output_name(args)  # append training args
    if args.ckpt:
        output_model_name = args.ckpt.split('--')[0]

    tags = [
        args.ptm_name, args.ptm_model_name, args.collection_conf, args.corpus,
        f's{args.seed}', f'e{args.epochs}', f'e{args.batch}', f'lr{args.lr}'
    ]
    if args.public:
        tags.append('public')
    if args.seed_only:
        tags.append('seed_labels')

    os.unsetenv('WANDB_API_KEY')
    params = {
        'job_type': 'llm_train',
        'name': output_model_name,
        'run_id': output_model_name + '@' + str(args.run_id),
        'run_group': args.collection_conf,
        'tags': tags,
        'conf': {
            'ptm_alias': args.ptm_name,
            'ptm': args.ptm_model_name,
            'lang': args.lang_conf,
            'corpus': args.corpus,
            'seed': args.seed,
            'epochs': args.epochs,
            'batch': args.batch,
            'learning_rate': args.lr
        }
    }

    return output_model_name, initialize_run(**params)


def __apply_peft(model_alias: str, model: PreTrainedModel, run: Any) -> Union[PreTrainedModel, PeftModel]:
    if model_alias not in __peft_confs:
        logger.warning(f'No PEFT config for [{model_alias}]')
        return model
    logger.info(f'Using peft config [{model_alias}] {__peft_confs}')
    peft_config = __peft_confs[model_alias]
    peft_model = get_peft_model(model, peft_config)
    if run is not None:
        run.config.update(peft_config)
    peft_model.print_trainable_parameters()

    return peft_model


def __get_optimizers(model_alias: str, model: PreTrainedModel, learning_rate: float) \
        -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    if 'llama' in model_alias:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
        return optimizer, scheduler

    return None, None


# noinspection DuplicatedCode
def llm_train_raw(args) -> int:
    """
    Train llm decoder models
    ./mulabel llm train --batch 8 --epochs 20 --lr 1e-4 --run_id 1 --ptm_name llama3b -c eurlex
    """
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

    model = AutoModelForCausalLM.from_pretrained(
        args.ptm_model_name,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        cache_dir=args.tmp_dir,
        use_cache=None,
        attn_implementation="sdpa",  # can be None; sdpa: use Flash Attention and Xformer memory-efficient kernels
        device_map=(
            "auto"
        ),
        torch_dtype=torch.bfloat16,
    )

    # model.to(device)

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        args.ptm_model_name
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logger.warning("Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    datasets, avg_k = construct_datasets(text_set, label_set, tokenizer, 512)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias='none'
    )
    peft_model = get_peft_model(model, peft_config)
    if run:
        run.config.update(peft_config)
    peft_model.print_trainable_parameters()

    peft_model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

    logger.info(f'Loaded train[{len(text_set["train"])}] dev[{len(text_set["dev"])}] test[{len(text_set["test"])}]')
    logger.info(f'Loaded {labeler} with {labeler.num_labels}')

    return 0


# noinspection DuplicatedCode
def llm_train(args) -> int:
    """
    Train llm decoder models
    ./mulabel llm train --batch 8 --epochs 20 --lr 1e-4 --run_id 1 --ptm_name llama3b -c eurlex
    """
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
        args.ptm_model_name, cache_dir=args.tmp_dir, num_labels=labeler.num_labels,
        id2label=labeler.ids_to_labels(), label2id=labeler.labels_to_ids(),
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        use_cache=None,
        attn_implementation="sdpa",  # can be None; sdpa: use Flash Attention and Xformer memory-efficient kernels
        device_map=(
            "auto"
        ),
        torch_dtype=torch.bfloat16,
        problem_type='multi_label_classification' if 'multilabel' == labeler.get_type_code()
        else 'single_label_classification'
    )

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        args.ptm_model_name
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logger.warning("Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    datasets, avg_k = construct_datasets(text_set, label_set, tokenizer, -1)

    model = __apply_peft(args.ptm_name, model, run)
    model.to(device)

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
        optimizers=(__get_optimizers(args.ptm_name, model, args.lr)),
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
    metrics.dump(result_path, {'seed': args.seed}, run)

    return 0
