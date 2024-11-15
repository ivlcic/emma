import os
import logging
from typing import Any, Tuple

import torch
import random

from argparse import ArgumentParser
from peft import get_peft_model, LoraConfig, TaskType

from torch import Tensor, optim
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from lightning import seed_everything

from ...core.dataset import TruncatedDataset
from ...core.args import CommonArguments
from ...core.models import llm_model_name_map
from ...core.metrics import Metrics
from ...core.wandb import initialize_run
from ..utils import __supported_languages, compute_arg_collection_name, compute_model_output_name, load_train_data

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

    model = AutoModelForCausalLM.from_pretrained(
        args.ptm_model_name,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        cache_dir=args.tmp_dir,
        use_cache=None,
        attn_implementation="sdpa",  # set this to None: use Flash Attention and Xformer memory-efficient kernels
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

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias='none'
    )
    model = get_peft_model(model, peft_config)
    if run:
        run.config.update(peft_config)
    model.print_trainable_parameters()

    model.to(device)

    datasets = {}
    average_labels_per_sample = 0
    for split in ['dev', 'test', 'train']:
        if split not in text_set.keys():
            continue
        datasets[split] = TruncatedDataset(text_set[split], label_set[split], tokenizer, 512)
        average_labels_per_sample += datasets[split].average_labels
    average_labels_per_sample /= 3
    avg_k = round(average_labels_per_sample)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

    logger.info(f'Loaded train[{len(text_set["train"])}] dev[{len(text_set["dev"])}] test[{len(text_set["test"])}]')
    logger.info(f'Loaded {labeler} with {labeler.num_labels}')

    return 0
