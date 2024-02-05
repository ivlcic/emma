import time
import numpy as np
import logging
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy, F1Score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers.optimization import get_linear_schedule_with_warmup


logger = logging.getLogger('core.classtrain')


# noinspection SpellCheckingInspection
class Classification(pl.LightningModule):

    valid_labels = ['binary', 'multilabel', 'multiclass']

    """
    Pytorch Lightning module to train all models
    """
    def __init__(self, model, lr, scheduler, label_type, chunk, num_labels, dataset_size, epochs, batch_size):
        super().__init__()
        if label_type not in Classification.valid_labels:
            raise RuntimeError(f'Invalid label type. Can be one of {Classification.valid_labels}')
        self.model = model
        self.lr = lr
        self.scheduler = scheduler
        self.label_type = label_type
        self.chunk = chunk
        self.num_labels = num_labels
        self.dataset_size = dataset_size
        self.epochs = epochs
        self.batch_size = batch_size
        if self.label_type == 'binary':
            self.eval_metric = Accuracy(task=self.label_type, num_classes=self.num_labels)
        elif self.label_type == 'multilabel':
            self.eval_metric = F1Score(task=self.label_type, num_classes=self.num_labels, average='micro')
        else:
            self.eval_metric = Accuracy(task=self.label_type, num_classes=self.num_labels, multiclass=True)

    def _log_step(self, prefix, y_pred, y_true, loss, start):
        self.log(
            prefix + 'eval_metric', self.eval_metric(y_pred, y_true),
            on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            prefix + 'loss', loss,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            prefix + 'time', time.time() - start,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def _compute_true_pred_loss(self, batch):
        if self.chunk:
            ids = [data['ids'] for data in batch]
            mask = [data['mask'] for data in batch]
            token_type_ids = [data['token_type_ids'] for data in batch]
            targets = [data['targets'][0] for data in batch]
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
            y_true = targets.to(self.device)

            y_hat = self.model(ids, mask, token_type_ids, length)
        else:
            ids = batch['ids'].to(self.device)
            mask = batch['mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            y_true = batch['labels'].to(self.device)

            y_hat = self.model(ids, mask, token_type_ids)

        if self.label_type == 'multilabel' or self.label_type == 'binary':
            loss = F.binary_cross_entropy_with_logits(y_hat, y_true.float())  # sigmoid + binary cross entropy loss
            y_pred = torch.sigmoid(y_hat)
        else:
            loss = F.cross_entropy(y_hat, y_true)  # softmax + cross entropy loss
            y_pred = torch.softmax(y_hat, dim=-1)
        return y_true, y_pred, loss

    def training_step(self, batch, batch_idx):
        start = time.time()
        metrics = {}
        y_true, y_pred, loss = self._compute_pred_true_loss(batch)

        metrics['loss'] = loss
        self._log_step('train_', y_pred, y_true, loss, start)
        return metrics

    def validation_step(self, batch, batch_idx, prefix='val_'):
        start = time.time()
        metrics = {}
        y_true, y_pred, loss = self._compute_pred_true_loss(batch)

        metrics[prefix + 'loss'] = loss
        metrics['preds'] = y_pred
        metrics['y'] = y_true

        self._log_step(prefix, y_pred, y_true, loss, start)
        return metrics

    def _logout_metrics(self, prefix: str, y_true, y_pred):
        logger.info("Epoch: %s", self.current_epoch)
        logger.info("%saccuracy: %.5f", prefix, accuracy_score(y_true, y_pred))
        for average_type in ['micro', 'macro', 'weighted']:
            if self.label_type == 'binary' and not average_type == 'macro':
                continue
            logger.info(
                '%s%s_precision: %.5f',
                prefix, average_type, precision_score(y_true, y_pred, average=average_type)
            )
            logger.info(
                '%s%s_recall: %.5f',
                prefix, average_type, recall_score(y_true, y_pred, average=average_type)
            )
            logger.info(
                '%s%s_recall: %.5f',
                prefix, average_type, f1_score(y_true, y_pred, average=average_type)
            )

    def validation_epoch_end(self, outputs, prefix='val_'):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output['y'].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output['preds'].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        y_pred = predictions.numpy()
        y_true = labels.numpy()

        if self.label_type == 'multilabel' or self.label_type == 'binary':
            y_pred_labels = np.where(y_pred > 0.5, 1, 0)
        else:
            y_pred_labels = np.argmax(y_pred, axis=1)

        self._logout_metrics(prefix, y_true, y_pred_labels)

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx, 'test_')
        return metrics

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs, prefix="test_")

    def configure_optimizers(self):
        opt = {}
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        opt['optimizer'] = optimizer
        if not self.scheduler:
            return opt
        else:
            num_steps = self.dataset_size * self.epochs / self.batch_size
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_steps * 0.1, num_training_steps=num_steps
            )
            opt['lr_scheduler'] = scheduler
            return opt
