import logging
import os
from argparse import ArgumentParser

from .twentynews import _prep as _prep_20news
from .booksummaries import _prep as _prep_booksummaries
from .hyperpartisan import _prep as _prep_hyperpartisan
from .eurlex import _prep as _prep_eurlex
from .utils import _download_file, _unzip_file, _remove_directory, _move_files, _write_csv
from ...core.args import CommonArguments

logger = logging.getLogger('longdoc.prep')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.split_data_dir(module_name, parser, ('-o', '--data_out_dir'))
    parser.add_argument(
        'corpora', help='Corpora files (prefix) to prep.', nargs='+',
        choices=['hyperpartisan', 'eurlex', '20news', 'booksummaries']
    )


def main(arg) -> int:
    logger.debug("Starting data preparation")
    for corpora in arg.corpora:
        split_dir = os.path.join(arg.data_out_dir, corpora)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        dl_dir = os.path.join(arg.data_in_dir, corpora)

        func = globals().get('_prep_' + corpora)
        func(dl_dir, split_dir)
    logger.debug("Finished data preparation")
    return 0
