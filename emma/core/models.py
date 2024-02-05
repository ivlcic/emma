from typing import Type, Optional

import torch
import logging
import numpy as np
import torch.nn.functional as func

from transformers import LongformerModel, PreTrainedModel, AutoTokenizer, PreTrainedTokenizer, AutoModel

from .dataset import TruncatedDataset, TruncatedPlusRandomDataset, TruncatedPlusTextRankDataset, ChunkDataset
from .labels import Labeler

logger = logging.getLogger('longdoc.prep.booksummaries')

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


class Module(torch.nn.Module):

    def __init__(self, model_name: str, labeler: Labeler, cache_model_dir: str, dev: Optional[str]):
        super().__init__()
        if dev is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device('cuda' if use_cuda else 'cpu')
            logger.info('Device was not set will use [%s].', device)
        else:
            device = torch.device(dev)
            logger.info('Device was set [%s] will use [%s].', dev, device)
        self.device = device
        self.model_name = model_name
        self.labeler = labeler
        self.cache_model_dir = cache_model_dir
        self.dataset_class = TruncatedDataset
        self.has_additional_text = False
        self.model = None
        self.tokenizer = None

    def get_max_len(self) -> int:
        return self.tokenizer.model_max_length

    def set_dataset_class(self, dataset_class: Type):
        self.dataset_class = dataset_class

    def get_dataset_class(self) -> Type:
        return self.dataset_class

    def set_has_additional_text(self, has_additional_text: bool):
        self.has_additional_text = has_additional_text


class TransEnc(Module):

    _allowed_models = ['xlm-roberta-base', 'bert-base-uncased', 'bert-base-multilingual-cased']

    def __init__(self, model_name: str, labeler: Labeler, cache_model_dir: str, device: str):
        super().__init__(model_name, labeler, cache_model_dir, device)
        if model_name not in TransEnc._allowed_models:
            raise RuntimeError(f'Only {TransEnc._allowed_models} are allowed!')
        self.model: PreTrainedModel = AutoModel.from_pretrained(model_name, cache_dir=cache_model_dir)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_model_dir)
        self.dropout = torch.nn.Dropout(0.0)
        self.classifier = torch.nn.Linear(768, len(self.labeler.labels))
        self.model.to(self.device)

    def set_dropout(self, dropout_rate: float):
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, ids, mask, token_type_ids):

        if not self.has_additional_text:
            _, model_output = self.model(
                ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
            )
            drop_output = self.dropout(model_output)
        else:
            _, truncated_output = self.model(
                ids[:, 0, :], attention_mask=mask[:, 0, :], token_type_ids=token_type_ids[:, 0, :], return_dict=False
            )
            _, additional_text_output = self.model(
                ids[:, 1, :], attention_mask=mask[:, 1, :], token_type_ids=token_type_ids[:, 1, :], return_dict=False
            )
            concat_output = torch.cat((truncated_output, additional_text_output), dim=1)  # batch_size, 768*2
            drop_output = self.dropout(concat_output)  # batch_size, 768*2

        logits = self.classifier(drop_output)
        return logits


class TransEncPlus(TransEnc):
    def __init__(self, model_name: str, labeler: Labeler, cache_model_dir: str, device: str):
        super().__init__(model_name, labeler, cache_model_dir, device)
        self.set_has_additional_text(True)
        self.set_dataset_class(TruncatedPlusRandomDataset)


class ToTransEncModel(Module):
    def __init__(self, model_name: str, labeler: Labeler, cache_model_dir: str, device: str):
        super().__init__(model_name, labeler, cache_model_dir, device)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_model_dir)
        self.trans = torch.nn.TransformerEncoderLayer(d_model=768, nhead=2)
        self.fc = torch.nn.Linear(768, 30)
        self.classifier = torch.nn.Linear(30, len(self.labeler.labels))
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
        self.classifier = LongformerClassificationHead(
            hidden_size=768, hidden_dropout_prob=0.1, num_labels=len(self.labeler.labels)
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_model_dir)
        self.dataset_class = TruncatedDataset

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


class LongformerClassificationHead(torch.nn.Module):
    # This class is from https://huggingface.co/transformers/_modules/transformers/models/longformer
    # /modeling_longformer.html#LongformerForSequenceClassification
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        # config from transformers.LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    # noinspection PyUnusedLocal
    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


class ModuleFactory:
    """Factory for creating model instances."""
    @staticmethod
    def get_module(model_name: str, labeler: Labeler, cache_model_dir: str, device: Optional[str]) -> Module:
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
        logger.info(
            'Loaded %s model with tokenizer %s for %s and max len %s.',
            module.model_name, module.tokenizer.name_or_path, module.dataset_class, module.get_max_len())
        return module
