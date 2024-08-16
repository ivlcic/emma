import numpy as np
import spacy
import pytextrank  # do not remove
import random
import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding


class TruncatedDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.max_len = max_len
        labels_per_sample = np.sum(labels, axis=1)
        self.average_labels = np.mean(labels_per_sample)
        self.std_labels = np.std(labels_per_sample, ddof=1)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        label_ids = torch.tensor(self.labels[index]).float().nonzero().squeeze().tolist()
        return {
            'input_ids': torch.tensor(ids),
            'attention_mask': torch.tensor(mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'labels': torch.tensor(self.labels[index]).float()
        }


class TruncatedPlusTextRankDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.max_len = max_len
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("textrank")
        labels_per_sample = np.sum(labels, axis=1)
        self.average_labels = np.mean(labels_per_sample)
        self.std_labels = np.std(labels_per_sample, ddof=1)

    def __len__(self):
        return len(self.text)

    # noinspection PyProtectedMember
    def apply_textrank(self, text):
        doc = self.nlp(text)
        num_phrases = len(list(doc._.phrases))
        num_sents = len(list(doc.sents))
        tr = doc._.textrank
        running_length = 0
        key_sents_idx = []
        key_sents = []
        for sentence in tr.summary(limit_phrases=num_phrases, limit_sentences=num_sents, preserve_order=False):
            if running_length <= (self.max_len - 2):
                sentence_str = str(sentence)
                sentence_tokens = self.tokenizer.tokenize(sentence_str)
                running_length += len(sentence_tokens)
                key_sents.append(sentence_str)
                key_sents_idx.append(sentence.sent.start)

        reorder_idx = list(np.argsort(key_sents_idx))
        selected_text = ''
        for idx in reorder_idx:
            selected_text += key_sents[idx] + ' '
        return selected_text

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True
        )

        if len(inputs['input_ids']) > 1:
            # select key sentences if text is longer than max length
            selected_text = self.apply_textrank(text)

            second_inputs = self.tokenizer.encode_plus(
                text=selected_text,
                text_pair=None,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_overflowing_tokens=True
            )
        else:
            second_inputs = inputs

        ids = (inputs['input_ids'][0], second_inputs['input_ids'][0])
        mask = (inputs['attention_mask'][0], second_inputs['attention_mask'][0])
        token_type_ids = (inputs["token_type_ids"][0], second_inputs["token_type_ids"][0])

        return {
            'input_ids': torch.tensor(ids),
            'attention_mask': torch.tensor(mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'labels': torch.tensor(self.labels[index]).float()
        }


class TruncatedPlusRandomDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.max_len = max_len
        self.nlp = spacy.load("en_core_web_sm")
        labels_per_sample = np.sum(labels, axis=1)
        self.average_labels = np.mean(labels_per_sample)
        self.std_labels = np.std(labels_per_sample, ddof=1)

    def __len__(self):
        return len(self.text)

    def select_random_sents(self, text):
        doc = self.nlp(text)
        sents = list(doc.sents)
        running_length = 0
        sent_idxs = list(range(len(sents)))
        selected_idx = []
        while running_length <= (self.max_len - 2) and sent_idxs:
            idx = random.choice(sent_idxs)
            sent_idxs.remove(idx)
            sentence = str(sents[idx])
            sentence_tokens = self.tokenizer.tokenize(sentence)
            running_length += len(sentence_tokens)
            selected_idx.append(idx)

        reorder_idx = sorted(selected_idx)
        selected_text = ''
        for idx in reorder_idx:
            selected_text += str(sents[idx]) + ' '
        return selected_text

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs: BatchEncoding = self.tokenizer(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            # return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True
        )
        if len(inputs['input_ids']) > 1:
            # select random sentences if text is longer than max length
            selected_text = self.select_random_sents(text)
            second_inputs = self.tokenizer(
                text=selected_text,
                text_pair=None,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                # return_tensors='pt',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_overflowing_tokens=True
            )
        else:
            second_inputs = inputs

        ids = (inputs['input_ids'][0], second_inputs['input_ids'][0])
        mask = (inputs['attention_mask'][0], second_inputs['attention_mask'][0])
        token_type_ids = (inputs['token_type_ids'][0], second_inputs['token_type_ids'][0])

        result = {
            'input_ids': torch.tensor(ids),
            'attention_mask': torch.tensor(mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'labels': torch.tensor(self.labels[index]).float()
        }
        return result


class ChunkDataset(Dataset):
    def __init__(self, text, labels, tokenizer, chunk_len=200, overlap_len=50):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.overlap_len = overlap_len
        self.chunk_len = chunk_len
        labels_per_sample = np.sum(labels, axis=1)
        self.average_labels = np.mean(labels_per_sample)
        self.std_labels = np.std(labels_per_sample, ddof=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = " ".join(str(self.text[index]).split())

        data = self.tokenizer(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.chunk_len,
            truncation=True,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True
        )

        ids = []
        mask = []
        token_type_ids = []
        targets = []
        for i, idx in enumerate(data['input_ids']):
            ids.append(torch.tensor(data['input_ids'][i], dtype=torch.long))
            mask.append(torch.tensor(data["attention_mask"][i], dtype=torch.long))
            token_type_ids.append(torch.tensor(data["token_type_ids"][i], dtype=torch.long))
            targets.append(torch.tensor(self.labels[index], dtype=torch.long).float())

        result = ({
            'input_ids': ids,
            'attention_mask': mask,
            'token_type_ids': token_type_ids,
            'labels': targets,
            'len': [torch.tensor(len(targets), dtype=torch.long)]
        })
        return result
