import logging

from argparse import ArgumentParser
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from .utils import _load_data, _chunk_collate_fn, _create_dataloader, _compute_output_name
from ...core.classtrain import Classification
from ...core.args import CommonArguments
from ...core.models import valid_model_names, ModuleFactory

logger = logging.getLogger('longdoc.prep')

corpora = {
    'hyperpartisan', 'eurlex57k', '20news', 'booksummaries'
}


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(module_name, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    CommonArguments.train(parser, 8)
    CommonArguments.num_workers(parser)
    parser.add_argument(
        '--corpus', help=f'Corpora files to train on {corpora}.'
    )
    parser.add_argument(
        '--model_name', type=str, required=True, help=f'Model name: {valid_model_names}'
    )
    parser.add_argument(
        '--ckpt', type=str,
        help='Path to a saved ckpt for continued training or evaluation'
             'e.g. bert_hyperpartisan_b8_e20_s3456_lr3e-05--epoch=17.ckpt'
    )


def main(args) -> int:
    if args.corpus not in corpora:
        raise RuntimeError(f'Corpus {args.corpus} should be one of {corpora}')

    logger.debug("Starting training")
    seed_everything(args.seed, workers=True)
    logger.debug("Loading data")

    text_set, label_set, labeler = _load_data(args.data_in_dir, args.corpus)
    logger.debug("Loaded data")
    module = ModuleFactory.get_module(
        args.model_name, labeler, args.tmp_dir, args.device
    )
    dataloaders = _create_dataloader(
        module, text_set, label_set, args.batch, args.num_workers
    )
    output_model_name = _compute_output_name(args)

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
    if args.ckpt:
        trainer = Trainer(
            logger=tb_logger,
            callbacks=[ckpt_config, device_stats],
            deterministic=True,
            num_sanity_val_steps=0,
            max_epochs=args.epochs,
            resume_from_checkpoint=args.data_out_dir + args.ckpt
        )
    else:
        trainer = Trainer(
            logger=tb_logger,
            callbacks=[ckpt_config, device_stats],
            deterministic=True,
            num_sanity_val_steps=0,
            max_epochs=args.epochs
        )

    logger.debug("Started training %s", output_model_name)
    trainer.fit(model=task, train_dataloaders=dataloaders['train'], val_dataloaders=dataloaders['dev'])

    logger.debug("Finished training")
    return 0
