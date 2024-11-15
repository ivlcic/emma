import os
from typing import Any, Dict, List, Optional

import pandas as pd


def initialize_run(job_type: str, name: str, run_group: str, run_id: str,
                   tags: List[str], conf: Dict[str, Any] = None) -> Any:
    if conf is None:
        conf = {}
    api_key = os.getenv('WANDB_API_KEY')
    if api_key is None:
        return None

    import wandb
    wandb.login('never', api_key)

    run = wandb.init(
        job_type,
        project=os.getenv('WANDB_PROJECT'),
        name=name,
        id=run_id,
        group=run_group,
        tags=tags,
        config=conf
    )
    return run


def send_metrics(run, artifact_name: str, files: List[str] = None):
    if files is None:
        files = []
    if run is None:
        return
    api_key = os.getenv('WANDB_API_KEY')
    if api_key is None:
        return
    import wandb
    artifact = wandb.Artifact(f'{artifact_name}', type='metrics')
    added = []
    for f in files:
        if os.path.exists(f) and os.path.isfile(f):
            artifact.add_file(f, overwrite=True)
            added.append(f)

    if len(added) > 0:
        run.log_artifact(artifact)
