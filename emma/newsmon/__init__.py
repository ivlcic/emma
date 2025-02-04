from ..core import ModuleDescriptor
from ..core.args import ModuleArguments, CommandArguments

MODULE_DESCRIPTOR = ModuleDescriptor(
    'newsmon',
    'Multilabel classification module',
    ModuleArguments([
        CommandArguments('prep', 'Prepares the data', multi_action=True),
        CommandArguments('fa', 'Faiss index tasks with lightweight predictor.', multi_action=True),
        CommandArguments('te', 'Transformer encoder tasks', multi_action=True)
    ])
)
