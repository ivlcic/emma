import time
import logging
import torch

import numpy as np
import lightning.pytorch as pl
import torch.nn.functional as funct
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from torchmetrics import Accuracy, F1Score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, ndcg_score
from transformers.optimization import get_linear_schedule_with_warmup

from .dataset import ChunkDataset
from .metrics import r_precision_at_k
from .models import Module

logger = logging.getLogger('core.class')


# noinspection SpellCheckingInspection
class Classification(pl.LightningModule):
    """
    Pytorch Lightning module to train all models
    """
    def __init__(self, module: Module, lr: float, scheduler: bool, dataset_size: int, epochs: int, batch_size: int):
        super().__init__()

        self.model = module
        self.lr = lr
        self.scheduler = scheduler
        self.label_type = module.labeler.get_type_code()
        self.chunk = module.get_dataset_class() == ChunkDataset
        self.num_labels = module.labeler.num_labels
        self.dataset_size = dataset_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_step_outputs = []
        self.test_step_outputs = []
        if self.label_type == 'binary':
            self.eval_metric = Accuracy(task=self.label_type, num_classes=self.num_labels)
        elif self.label_type == 'multilabel':
            self.eval_metric = F1Score(task=self.label_type, num_labels=self.num_labels, average='micro')
        else:
            self.eval_metric = Accuracy(task=self.label_type, num_classes=self.num_labels, average='macro')

    def _log_step(self, prefix, y_pred, y_true, loss, start):
        self.log(
            prefix + 'eval_metric', self.eval_metric(y_pred, y_true),
            on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
        self.log(
            prefix + 'loss', loss,
            on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
        self.log(
            prefix + 'time', time.time() - start,
            on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )

    def _compute_true_pred_loss(self, batch):
        if self.chunk:
            ids = [data['input_ids'] for data in batch]
            mask = [data['attention_mask'] for data in batch]
            token_type_ids = [data['token_type_ids'] for data in batch]
            targets = [data['labels'][0] for data in batch]
            length = [data['len'] for data in batch]

            ids = torch.cat(ids)
            mask = torch.cat(mask)
            token_type_ids = torch.cat(token_type_ids)
            targets = torch.stack(targets)
            length = torch.cat(length)
            length = [x.item() for x in length]

            ids = ids.to(self.device)
            mask = mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            labels = targets.to(self.device)

            logits = self.model(ids, mask, token_type_ids, length)
        else:
            ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits = self.model(ids, mask, token_type_ids)

        if self.label_type == 'multilabel':
            loss_fct = BCEWithLogitsLoss()  # sigmoid + binary cross entropy loss
            loss = loss_fct(logits, labels)
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # softmax + cross entropy loss
            prob = torch.softmax(logits, dim=-1)
            _, pred = torch.max(prob, 1)
        return labels, pred, loss, prob

    def training_step(self, batch, batch_idx):
        start = time.time()
        metrics = {}
        labels, pred, loss, prob = self._compute_true_pred_loss(batch)

        metrics['loss'] = loss
        self._log_step('train_', pred, labels, loss, start)
        return metrics

    def validation_step(self, batch, batch_idx, prefix='val_'):
        start = time.time()
        outputs = {}
        labels, pred, loss, prob = self._compute_true_pred_loss(batch)

        outputs[prefix + 'loss'] = loss
        outputs['preds'] = pred
        outputs['prob'] = prob
        outputs['y'] = labels

        self._log_step(prefix, pred, labels, loss, start)
        if prefix == 'val_':
            self.validation_step_outputs.append(outputs)
        else:
            self.test_step_outputs.append(outputs)
        return outputs

    def _logout_metrics(self, prefix: str, y_true, y_pred, y_prob, k: int = 5):
        self.model.logger.info(f'Epoch: {self.current_epoch}')
        self.model.logger.info(f'{prefix}accuracy: %.7f', accuracy_score(y_true, y_pred))
        for average_type in ['micro', 'macro', 'weighted']:
            if self.label_type == 'binary' and not average_type == 'macro':
                continue
            self.model.logger.info(
                f'{prefix}{average_type}_precision: %.7f', precision_score(y_true, y_pred, average=average_type)
            )
            self.model.logger.info(
                f'{prefix}{average_type}_recall: %.7f', recall_score(y_true, y_pred, average=average_type)
            )
            self.model.logger.info(
                f'{prefix}{average_type}_f1: %.7f', f1_score(y_true, y_pred, average=average_type)
            )
        if self.label_type == 'multilabel':
            self.model.logger.info(f'{prefix}hamming_loss: %.7f', hamming_loss(y_true, y_pred))
            self.model.logger.info(f'{prefix}ndcg@%s: %.7f', k, ndcg_score(y_true, y_prob, k=k))
            self.model.logger.info(f'{prefix}ndcg: %.7f', k, ndcg_score(y_true, y_prob))
            self.model.logger.info(f'{prefix}r-precision@%s: %.7f', k, r_precision_at_k(y_true, y_prob, k=k))

    def _validation_epoch_end(self, outputs, prefix='val_'):
        labels = []
        predictions = []
        probabilities = []
        for output in outputs:
            for out_labels in output['y'].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output['preds'].detach().cpu():
                predictions.append(out_predictions)
            for out_probabilities in output['prob'].detach().cpu():
                probabilities.append(out_probabilities)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        probabilities = torch.stack(probabilities)

        y_pred = predictions.numpy()
        y_true = labels.numpy()
        y_prob = probabilities.numpy()

        self._logout_metrics(
            prefix, y_true, y_pred, y_prob, round(self.model.average_labels_per_sample)
        )

    def on_validation_epoch_end(self):
        self._validation_epoch_end(self.validation_step_outputs)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        outputs = self.validation_step(batch, batch_idx, 'test_')
        return outputs

    def on_test_epoch_end(self):
        self._validation_epoch_end(self.test_step_outputs, "test_")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        opt = {}
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        opt['optimizer'] = optimizer
        if not self.scheduler:
            return opt
        else:
            num_steps = self.dataset_size * self.epochs / self.batch_size
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(num_steps * 0.1), num_training_steps=int(num_steps)
            )
            opt['lr_scheduler'] = scheduler
            return opt
