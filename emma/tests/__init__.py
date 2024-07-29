from ..core import ModuleDescriptor
from ..core.args import ModuleArguments, CommandArguments

MODULE_DESCRIPTOR = ModuleDescriptor(
    'tests',
    'Tests module',
    ModuleArguments([
        CommandArguments('llm', 'LLM test tasks.', multi_action=True)
    ])
)

