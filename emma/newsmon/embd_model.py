import os
from typing import Union, List, Dict

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel

from .gte import GTEEmbedding

retrieve_model_name_map = {
    'bge_m3': 'BAAI/bge-m3',
    'jina3': 'jinaai/jina-embeddings-v3',
    'gte': 'Alibaba-NLP/gte-multilingual-base',
    'kalm_v15': 'HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5',
    'n_bge_m3': 'newsmon_bge-m3',
    'e_bge_m3': 'eurlex_bge-m3',
}


class EmbeddingModelWrapper(object):
    def __init__(self, short_name: str, name: str):
        self.short_name = short_name
        self.name = name

    def embed(self, text_to_embed: Union[str, List[str]]) -> np.ndarray:
        pass


class Jina3(EmbeddingModelWrapper):

    def __init__(self):
        super(Jina3, self).__init__('jina3', 'jinaai/jina-embeddings-v3')
        self.model = AutoModel.from_pretrained(
            self.name, trust_remote_code=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)


    def embed(self, text_to_embed: Union[str, List[str]]) -> np.ndarray:
        return self.model.encode(text_to_embed, task='text-matching', show_progress_bar=False)


class Gte(EmbeddingModelWrapper):

    def __init__(self):
        super(Gte, self).__init__('gte', 'Alibaba-NLP/gte-multilingual-base')
        self.model = GTEEmbedding(self.name)

    def embed(self, text_to_embed: Union[str, List[str]]) -> np.ndarray:
        # noinspection PyTypeChecker
        return self.model.encode(text_to_embed)['dense_embeddings']


class BgeM3(EmbeddingModelWrapper):

    def __init__(self, short_name: Union[str, None] = None, path_or_name: Union[str, None] = None):
        if path_or_name is None:
            path_or_name = 'BAAI/bge-m3'
        if short_name is None:
            short_name = 'bge_m3'
        if not os.path.exists(path_or_name):
            super(BgeM3, self).__init__(short_name, path_or_name)
        else:
            name = os.path.basename(path_or_name)
            super(BgeM3, self).__init__(short_name, name)
        self.model = BGEM3FlagModel(
            path_or_name, use_fp16=True, device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    def embed(self, text_to_embed: Union[str, List[str]]) -> np.ndarray:
        return self.model.encode(text_to_embed)['dense_vecs']


class EmbeddingModelWrapperFactory:
    @classmethod
    def init_models(cls, args) -> Dict[str, EmbeddingModelWrapper]:
        if 'ptm_models' not in args or not args.ptm_models:
            args.ptm_models = retrieve_model_name_map.keys()
        else:
            args.ptm_models = args.ptm_models.split(',')
        models = {}
        for ptm_name, name in retrieve_model_name_map.items():
            if ptm_name not in args.ptm_models:
                continue
            if ptm_name == 'bge_m3':
                models[ptm_name] = BgeM3(ptm_name)
            if ptm_name == 'n_bge_m3':
                models[ptm_name] = BgeM3(ptm_name, str(os.path.join(args.data_result_dir, 'test', name)))
            if ptm_name == 'e_bge_m3':
                models[ptm_name] = BgeM3(ptm_name, str(os.path.join(args.data_result_dir, 'test', name)))
            if ptm_name == 'jina3':
                models[ptm_name] = Jina3()
            if ptm_name == 'gte':
                models[ptm_name] = Gte()
        return models
