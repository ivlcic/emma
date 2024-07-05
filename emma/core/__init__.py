from __future__ import annotations

import importlib
import logging.config
import os
import re

from typing import TypeVar
from colorlog import ColoredFormatter
from colorlog.formatter import ColoredRecord

from .args import ModuleArguments, ArgumentParser


class CoreColoredFormatter(ColoredFormatter):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def formatMessage(self, record: logging.LogRecord) -> str:
        """Format a message from a record object."""
        escapes = self._escape_code_map(record.levelname)
        # record.msg = record.msg.replace('[%s]', '[\033[93m%s\033[0m]')
        wrapper = ColoredRecord(record, escapes)
        wrapper.levelname = '[%s]' % wrapper.levelname
        wrapper.funcName = '[%s]' % wrapper.funcName
        wrapper.lineno = '[\033[37m%s\033[0m]' % wrapper.lineno
        wrapper.msg = wrapper.msg.replace('[%s]', '\033[93m%s\033[0m')
        wrapper.message = re.sub(r'\[([^]]+)]', r'[\033[93m\1\033[0m]', wrapper.message)
        message = logging.Formatter.formatMessage(self, wrapper)  # type: ignore
        message = self._append_reset(message, escapes)
        return message


LOGGING = {
    'version': 1,
    'formatters': {
        'my_formatter': {
            '()': CoreColoredFormatter,
            'datefmt': '%Y-%m-%d %H:%M:%S',
            'format': 's%(asctime)s '
                      '%(log_color)s%(levelname)-7s%(reset)s%(reset)s '
                      '%(yellow)s%(name)s%(reset)s %(lineno)-3s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'my_formatter',
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    },
}

logging.config.dictConfig(LOGGING)

logger = logging.getLogger('core')


class ModuleDescriptor:

    project = os.path.basename(os.path.dirname(os.path.dirname(__file__)))

    def __init__(self, name: str, descr: str, arg: ModuleArguments):
        self._name = name
        self._description = descr
        self._args = arg
        self._args.init_parser(
            ModuleDescriptor.project, name, descr
        )

    def get_name(self) -> str:
        return self._name

    def get_description(self) -> str:
        return self._description

    def get_args(self) -> ModuleArguments:
        return self._args


TExecModule = TypeVar('TExecModule', bound='ExecModule')


class ExecModule:

    @classmethod
    def get(cls, file_as_module_name: str) -> TExecModule:
        package = os.path.splitext(os.path.basename(file_as_module_name))[0]
        logger.debug('Loading package: [%s]', package)
        py_module = importlib.import_module(ModuleDescriptor.project + '.' + package)
        logger.debug('Imported package: [%s]', package)
        m: ExecModule = ExecModule(py_module)
        logger.debug('Loaded module: [%s]', package)
        return m

    def __init__(self, py_module):
        self._py_module = py_module
        self._descriptor: ModuleDescriptor = py_module.MODULE_DESCRIPTOR

    def execute(self) -> int:
        module_args: ModuleArguments = self._descriptor.get_args()
        parser: ArgumentParser = module_args.get_parser()
        arg = parser.parse_args()
        if arg.debug:
            logging.getLogger('root').setLevel(logging.DEBUG)

        action = arg.action.replace('-', '_')
        pym_name = ModuleDescriptor.project + '.' + self._descriptor.get_name() + '.' + action
        logger.debug('Loading Python module: [%s]', pym_name)
        py_module = importlib.import_module(pym_name)

        arg.func = None
        arg.module_name = self._descriptor.get_name()
        if hasattr(arg, 'sub_action') and arg.sub_action is not None:
            sub_action = arg.sub_action.replace('-', '_')
            fn = getattr(py_module, arg.action + '_' + sub_action, None)
            if fn is not None:
                arg.func = fn
        if arg.func is None:
            arg.func = py_module.main
        return arg.func(arg)
