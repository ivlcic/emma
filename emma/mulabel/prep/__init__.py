import logging
from argparse import ArgumentParser
from datetime import datetime, timezone, timedelta

from transformers import AutoTokenizer

from ...core.args import CommonArguments

logger = logging.getLogger('mulabel.prep')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.tmp_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_out_dir'))
    beginning_of_day = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    parser.add_argument(
        '-s', '--start_date', help='Articles start selection date.', type=str,
        default=beginning_of_day.astimezone(timezone.utc).isoformat()
    )
    next_day = beginning_of_day + timedelta(days=1)
    parser.add_argument(
        '-e', '--end_date', help='Articles end selection date.', type=str,
        default=next_day.astimezone(timezone.utc).isoformat()
    )
    parser.add_argument(
        '-c', '--country', help='Articles selection country.', type=str
    )


def prep_gather_labels(arg) -> int:
    """
    ./mulabel prep gather_labels -i /home/nikola/projects/neuroticla/result/corpus -s 2023-12-02 -e 2023-12-03
    """
    logger.debug("Starting data gathering to simplify format.")
    cache_dir = CommonArguments._package_path('tmp', 'prep')


def prep_gather(arg) -> int:
    """
    ./mulabel prep gather -i /home/nikola/projects/neuroticla/result/corpus -s 2023-12-02 -e 2023-12-03
    """
    logger.debug("Starting data gathering to simplify format.")
    cache_dir = CommonArguments._package_path('tmp', 'prep')
    deb_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/mdeberta-v3-base", cache_dir=cache_dir
    )  # 190M, CC100
    return 0


def prep_analyse(arg) -> int:
    pass
