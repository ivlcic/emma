from ..core import ModuleDescriptor
from ..core.args import ModuleArguments, CommandArguments

MODULE_DESCRIPTOR = ModuleDescriptor(
    'mulabel',
    'Multilabel classification module',
    ModuleArguments([
        CommandArguments('db', 'Deprecated Vector database tasks.', multi_action=True),
        CommandArguments('prep', 'Prepares the data', multi_action=True),
        CommandArguments('es', 'Elasticsearch database tasks.', multi_action=True),
        CommandArguments('fa', 'Faiss index tasks with lightweight predictor.', multi_action=True),
        CommandArguments('te', 'Transformer encoder tasks', multi_action=True),
        CommandArguments('llm', 'LLM tasks', multi_action=True)
    ])
)
