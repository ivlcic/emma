from ..core import ModuleDescriptor
from ..core.args import ModuleArguments, CommandArguments

MODULE_DESCRIPTOR = ModuleDescriptor(
    'longdoc',
    'Efficient Classification of Long Documents Using Transformers',
    ModuleArguments([
        CommandArguments('prep', 'Prepares the data.'),
        CommandArguments('train', 'Trains the model with Lightning.'),
        CommandArguments('hf_train', 'Trains the model with HuggingFace train.'),
        #CommandArguments('test', 'Unit test the model'),
        #CommandArguments('infer', 'Infer the model'),
        #CommandArguments('unittest', 'Internal tests')
    ])
)
