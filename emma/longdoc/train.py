import os
import random
import torch
import logging

from argparse import ArgumentParser

from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from .ds_utils import _load_data, _create_dataloader, _compute_output_name, _get_long_texts_and_labels
from ..core.classification import Classification
from ..core.args import CommonArguments
from ..core.models import valid_model_names, ModuleFactory

logger = logging.getLogger('longdoc.prep')

corpora = {
    'hyperpartisan', 'eurlex', 'eurlexinv', '20news', 'booksummaries', 'klp1s0'
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


def main(args) -> int:
    """
    ./longdoc train --batch 8 --epochs 20 --lr 5e-5 --model_name bertmc --corpus eurlex
    ./longdoc train --batch 8 --epochs 20 --lr 5e-5 --model_name xlmrb --corpus eurlex

    ./longdoc train --batch 8 --epochs 20 --lr 5e-5 --model_name tobertmc --corpus eurlex
    ./longdoc train --batch 8 --epochs 20 --lr 5e-5 --model_name toxlmrb --corpus eurlex

    ./longdoc train --batch 8 --epochs 20 --lr 5e-5 --model_name bertmcplusrandom --corpus eurlex
    ./longdoc train --batch 8 --epochs 20 --lr 5e-5 --model_name xlmrbplusrandom --corpus eurlex
    """
    if args.corpus not in corpora:
        raise RuntimeError(f'Corpus {args.corpus} should be one of {corpora}')
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision('high')
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

    module = ModuleFactory.get_module(
        args.model_name, labeler, args.tmp_dir, args.device, output_model_name
    )
    module.set_dropout(args.dropout)
    dataloaders = _create_dataloader(
        module, text_set, label_set, args.batch, args.num_workers
    )

    long_text_set, long_label_set = _get_long_texts_and_labels(text_set, label_set, module.tokenizer)
    long_dataloaders = _create_dataloader(
        module, long_text_set, long_label_set, args.batch, args.num_workers
    )

    dataset_size = len(label_set['train'])  # to calculate the num of steps for warm up scheduler
    logger.debug("Constructing task")
    task = Classification(module, args.lr, args.scheduler, dataset_size, args.epochs, args.batch)

    ckpt_config = ModelCheckpoint(
        monitor="val_eval_metric_epoch",
        verbose=False,
        save_top_k=1,
        save_weights_only=False,
        mode='max',
        every_n_epochs=1,
        dirpath=args.data_out_dir,
        filename=output_model_name + "--{epoch}"
    )
    tb_logger = TensorBoardLogger("tb_logs", name=output_model_name)
    device_stats = DeviceStatsMonitor()

    trainer = Trainer(
        logger=tb_logger,
        callbacks=[ckpt_config, device_stats],
        deterministic=True,
        num_sanity_val_steps=0,
        max_epochs=args.epochs
    )

    logger.debug("Starting training %s", output_model_name)
    if args.ckpt:
        args.ckpt = str(os.path.join(args.data_out_dir, args.ckpt))
        task = Classification.load_from_checkpoint(
            args.ckpt,
            module=module,
            lr=args.lr,
            scheduler=args.scheduler,
            dataset_size=dataset_size,
            epochs=args.epochs,
            batch_size=args.batch
        )

    trainer.fit(
        model=task,
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['dev'],
        ckpt_path=args.ckpt
    )

    for _ckpt in range(len(trainer.checkpoint_callbacks)):
        paths = trainer.checkpoint_callbacks[_ckpt]
        ckpt_path = trainer.checkpoint_callbacks[_ckpt].best_model_path
        module.logger.info("Testing with checkpoint path: {}".format(ckpt_path))
        metrics = trainer.test(dataloaders=dataloaders['test'], ckpt_path=ckpt_path)
        for metric in metrics:
            for key in metric:
                module.logger.info("%s: %.5f", key, metric[key])

        module.logger.info("Evaluating on long documents.")
        for split in ['dev', 'test']:
            module.logger.info("Evaluating on long documents in the [%s] set only", split)
            metrics = trainer.test(dataloaders=long_dataloaders[split], ckpt_path=ckpt_path)
            for metric in metrics:
                for key in metric:
                    module.logger.info("long_%s_%s: %.5f", split, key, metric[key])

    logger.debug("Finished training")
    return 0
