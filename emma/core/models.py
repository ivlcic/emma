import os.path
from typing import Type, Optional

import torch
import logging
import numpy as np
import torch.nn.functional as func

from transformers import LongformerModel, PreTrainedModel, AutoTokenizer, PreTrainedTokenizer, AutoModel

from .dataset import TruncatedDataset, TruncatedPlusRandomDataset, TruncatedPlusTextRankDataset, ChunkDataset
from .labels import Labeler

logger = logging.getLogger('core.models')

valid_model_names = {
    'bert', 'bertplustextrank', 'bertplusrandom', 'tobert',
    'bertmc', 'bertmcplustextrank', 'bertmcplusrandom', 'tobertmc',
    'xlmrb', 'xlmrbplustextrank', 'xlmrbplusrandom', 'toxlmrb',
    'longformer'
}

model_name_map = {
    'bert': 'bert-base-uncased',
    'bertmc': 'bert-base-multilingual-cased',
    'xlmrb': 'xlm-roberta-base',
    'longformer': 'allenai/longformer-base-4096'
}

llm_model_name_map = {
    'llama1b': 'meta-llama/Llama-3.2-1B',
    'llama3b': 'meta-llama/Llama-3.2-3B'
}

retrieve_model_name_map = {
    'bge_m3': 'BAAI/bge-m3',
    'jina3': 'jinaai/jina-embeddings-v3',
    'gte': 'Alibaba-NLP/gte-multilingual-base',
    'kalm_v15': 'HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5'
}


class Module(torch.nn.Module):

    @staticmethod
    def construct_logger(logger_name: str, logger_dir: Optional[str]) -> logging.Logger:
        m_logger = logging.getLogger(logger_name)
        m_logger.setLevel(logging.DEBUG)
        m_formatter = logging.Formatter(
            '%(asctime)s %(levelname)-7s %(name)s %(lineno)-3s: %(message)s', '%Y-%m-%d %H:%M:%S'
        )
        if not logger_dir:
            m_file_handler = logging.FileHandler(logger_name + '.log')
        else:
            if not os.path.exists(logger_dir):
                os.makedirs(logger_dir)
            m_file_handler = logging.FileHandler(os.path.join(logger_dir, logger_name + '.log'))
        m_file_handler.setLevel(logging.INFO)
        m_file_handler.setFormatter(m_formatter)
        m_logger.addHandler(m_file_handler)
        return m_logger

    def __init__(self, model_name: str, labeler: Labeler, cache_model_dir: str, dev: Optional[str]):
        super().__init__()
        if dev is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device('cuda' if use_cuda else 'cpu')
            logger.info('Device was not set will use [%s].', device)
        else:
            device = torch.device(dev)
            logger.info('Device was set [%s] will use [%s].', dev, device)
        self.logger = logger
        self.device = device
        self.model_name = model_name
        self.labeler = labeler
        self.cache_model_dir = cache_model_dir
        self.dataset_class = TruncatedDataset
        self.has_additional_text = False
        self.dropout_rate = 0.0
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.hidden_size = 768  # must be set when extending
        self.model = None
        self.tokenizer = None
        self.average_labels_per_sample = 0

    def get_max_len(self) -> int:
        return self.tokenizer.model_max_length

    def set_dataset_class(self, dataset_class: Type):
        self.dataset_class = dataset_class

    def set_average_labels_per_sample(self, average_labels_per_sample: float):
        self.average_labels_per_sample = average_labels_per_sample

    def get_dataset_class(self) -> Type:
        return self.dataset_class

    def set_has_additional_text(self, has_additional_text: bool):
        self.has_additional_text = has_additional_text

    def set_dropout(self, dropout_rate: float):
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.__init_classifier()


class XlmClassificationHead(torch.nn.Module):

    def __init__(self, hidden_size: int, dropout: torch.nn.Dropout, num_labels: int):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BertClassificationHead(torch.nn.Module):

    def __init__(self, hidden_size: int, num_labels: int, dropout: torch.nn.Dropout):
        super().__init__()
        self.dropout = dropout
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.out_proj(x)
        return x


class TransEnc(Module):

    _allowed_models = ['xlm-roberta-base', 'bert-base-uncased', 'bert-base-multilingual-cased']

    def __init__(self, model_name: str, labeler: Labeler, cache_model_dir: str, device: str):
        super().__init__(model_name, labeler, cache_model_dir, device)
        if model_name not in TransEnc._allowed_models:
            raise RuntimeError(f'Only {TransEnc._allowed_models} are allowed!')
        self.model: PreTrainedModel = AutoModel.from_pretrained(model_name, cache_dir=cache_model_dir)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_model_dir)
        self.__init_classifier()
        self.model.to(self.device)

    def __init_classifier(self):
        if 'xlm' in self.model_name:
            self.classifier = XlmClassificationHead(
                hidden_size=self.hidden_size, dropout=self.dropout, num_labels=self.labeler.num_labels
            )
        else:
            self.classifier = BertClassificationHead(
                hidden_size=self.hidden_size, dropout=self.dropout, num_labels=self.labeler.num_labels
            )

    def forward(self, input_ids, attention_mask, token_type_ids):

        if not self.has_additional_text:
            sequence_output, pooled_output = self.model(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False
            )
            if 'xlm' in self.model_name:
                model_output = sequence_output
            else:
                model_output = pooled_output
        else:
            sequence_output, pooled_output = self.model(
                input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :], token_type_ids=token_type_ids[:, 0, :],
                return_dict=False
            )
            additional_sequence_output, additional_pooled_output = self.model(
                input_ids[:, 1, :], attention_mask=attention_mask[:, 1, :], token_type_ids=token_type_ids[:, 1, :],
                return_dict=False
            )
            if 'xlm' in self.model_name:
                model_output = torch.cat((sequence_output, additional_sequence_output), dim=1)  # batch_size, 768*2
            else:
                model_output = torch.cat((pooled_output, additional_pooled_output), dim=1)  # batch_size, 768*2

        logits = self.classifier(model_output)
        return logits


class TransEncPlus(TransEnc):
    def __init__(self, model_name: str, labeler: Labeler, cache_model_dir: str, device: str):
        super().__init__(model_name, labeler, cache_model_dir, device)
        self.hidden_size = self.hidden_size * 2
        self.set_has_additional_text(True)
        self.set_dataset_class(TruncatedPlusRandomDataset)
        self.__init_classifier()


class ToTransEncModel(Module):
    _allowed_models = ['bert-base-uncased', 'bert-base-multilingual-cased']
    def __init__(self, model_name: str, labeler: Labeler, cache_model_dir: str, device: str):
        super().__init__(model_name, labeler, cache_model_dir, device)
        if model_name not in ToTransEncModel._allowed_models:
            raise RuntimeError(f'Only {ToTransEncModel._allowed_models} are allowed!')

        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_model_dir)
        #TODO: works only for bert type of models
        self.trans = torch.nn.TransformerEncoderLayer(d_model=768, nhead=2)
        self.fc = torch.nn.Linear(768, 30)
        self.classifier = torch.nn.Linear(30, self.labeler.num_labels)
        self.dataset_class = ChunkDataset
        self.model.to(self.device)

    def forward(self, ids, mask, token_type_ids, length):
        _, pooled_out = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        chunks_emb = pooled_out.split_with_sizes(length)
        batch_emb_pad = torch.nn.utils.rnn.pad_sequence(
            chunks_emb, padding_value=0, batch_first=True)
        batch_emb = batch_emb_pad.transpose(0, 1)  # (B,L,D) -> (L,B,D)
        padding_mask = np.zeros([batch_emb.shape[1], batch_emb.shape[0]])  # Batch size, Sequence length
        for idx in range(len(padding_mask)):
            padding_mask[idx][length[idx]:] = 1  # padding key = 1 ignored

        padding_mask = torch.tensor(padding_mask).to(self.device, dtype=torch.bool)
        trans_output = self.trans(batch_emb, src_key_padding_mask=padding_mask)
        mean_pool = torch.mean(trans_output, dim=0)  # Batch size, 768
        fc_output = self.fc(mean_pool)
        relu_output = func.relu(fc_output)
        logits = self.classifier(relu_output)

        return logits


class LongformerClass(Module):
    def __init__(self, model_name: str, labeler: Labeler, cache_model_dir: str, device: str):
        super().__init__(model_name, labeler, cache_model_dir, device)
        self.longformer = LongformerModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            gradient_checkpointing=True,
            cache_dir=cache_model_dir
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_model_dir)
        self.dataset_class = TruncatedDataset
        self.__init_classifier()

    def __init_classifier(self):
        self.classifier = XlmClassificationHead(
            hidden_size=self.hidden_size, dropout=self.dropout, num_labels=self.labeler.num_labels
        )

    def forward(self, ids, mask, token_type_ids):
        # Initialize global attention on CLS token
        global_attention_mask = torch.zeros_like(ids)
        global_attention_mask[:, 0] = 1
        sequence_output, _ = self.longformer(
            ids,
            attention_mask=mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        logits = self.classifier(sequence_output)
        return logits


class ModuleFactory:
    """Factory for creating model instances."""
    @staticmethod
    def get_module(model_name: str, labeler: Labeler, cache_model_dir: str, device: Optional[str],
                   log: Optional[str]) -> Module:
        if model_name not in valid_model_names:
            raise RuntimeError(f'Model {model_name} should be one of {valid_model_names}!')
        module: Module
        if 'plustextrank' in model_name:
            logger.info('Loading %s model for %s.', TransEncPlus, model_name)
            pretrain_name = model_name[:-len('plustextrank')]
            module = TransEncPlus(model_name_map[pretrain_name], labeler, cache_model_dir, device)
            module.set_dataset_class(TruncatedPlusTextRankDataset)
        elif 'plusrandom' in model_name:
            logger.info('Loading %s model for %s.', TransEncPlus, model_name)
            pretrain_name = model_name[:-len('plusrandom')]
            module = TransEncPlus(model_name_map[pretrain_name], labeler, cache_model_dir, device)
        elif model_name.startswith('to'):
            logger.info('Loading %s model for %s.', ToTransEncModel, model_name)
            pretrain_name = model_name[2:]
            module = ToTransEncModel(model_name_map[pretrain_name], labeler, cache_model_dir, device)
        elif model_name == 'longformer':
            logger.info('Loading %s model for %s.', LongformerClass, model_name)
            pretrain_name = model_name
            module = LongformerClass(model_name_map[pretrain_name], labeler, cache_model_dir, device)
        else:
            logger.info('Loading %s model for %s.', TransEnc, model_name)
            module = TransEnc(model_name_map[model_name], labeler, cache_model_dir, device)

        if log:
            module.logger = Module.construct_logger(log, None)

        logger.info(
            'Loaded %s model with tokenizer %s for %s and max len %s.',
            module.model_name, module.tokenizer.name_or_path, module.dataset_class, module.get_max_len())
        return module
