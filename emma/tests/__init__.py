from ..core import ModuleDescriptor
from ..core.args import ModuleArguments, CommandArguments

MODULE_DESCRIPTOR = ModuleDescriptor(
    'tests',
    'Tests module',
    ModuleArguments([
        CommandArguments('llm', 'LLM test tasks.', multi_action=True),
        CommandArguments('ir_metrics', 'IR Metrics tests.', multi_action=True),
        CommandArguments('label_assign', 'Label Assignment tests.', multi_action=True)
    ])
)

