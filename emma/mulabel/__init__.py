from ..core import ModuleDescriptor
from ..core.args import ModuleArguments, CommandArguments

MODULE_DESCRIPTOR = ModuleDescriptor(
    'mulabel',
    'Multilabel classification module',
    ModuleArguments([
        CommandArguments('prep', 'Prepares the data'),
        CommandArguments('split', 'Splits the data'),
        CommandArguments('train', 'Trains the model'),
        CommandArguments('test', 'Unit test the model'),
        CommandArguments('infer', 'Infer the model'),
        CommandArguments('unittest', 'Internal tests')
    ])
)
