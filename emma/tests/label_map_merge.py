import logging
from argparse import ArgumentParser

from ..core.args import CommonArguments

logger = logging.getLogger('tests.label_map_merge')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_in_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-i', '--data_out_dir'))


def label_map_merge(args) -> int:

    return 0
