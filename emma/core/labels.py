import logging

from abc import ABC, abstractmethod
from typing import List, Optional, Any

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, LabelEncoder

logger = logging.getLogger('core.labels')


class Labeler(ABC):

    def __init__(self, encoder, labels: Optional[List] = None):
        self.encoder = encoder
        self.labels = set(labels) if labels else set()
        self.computed = False

    def collect(self, labels: List[Any]):
        for x in labels:
            if isinstance(x, (list, set, tuple)):
                self.labels.update(x)
            else:
                self.labels.add(x)

    def fit(self):
        self.computed = True
        logger.debug('Total number of labels: %s', len(self.labels))

    @abstractmethod
    def get_type_code(self):
        pass

    def vectorize(self, labels: List) -> List:
        if not self.computed:
            self.fit()

        result = self.encoder.transform(labels)
        return result


class BinaryLabeler(Labeler):
    def __init__(self, labels: Optional[List] = None):
        super().__init__(LabelBinarizer(), labels)

    def fit(self):
        self.encoder.fit(list(self.labels))
        super().fit()

    @classmethod
    def type_code(cls) -> str:
        return 'binary'

    def get_type_code(self):
        type(self).type_code()


class MulticlassLabeler(Labeler):

    def __init__(self, labels: Optional[List] = None):
        super().__init__(LabelEncoder(), labels)

    def fit(self):
        self.encoder.fit(list(self.labels))
        super().fit()

    @classmethod
    def type_code(cls) -> str:
        return 'multiclass'

    def get_type_code(self):
        type(self).type_code()


class MultilabelLabeler(Labeler):

    def __init__(self, labels: Optional[List] = None):
        super().__init__(MultiLabelBinarizer(), labels)

    def fit(self):
        self.encoder.fit([list(self.labels)])
        super().fit()

    @classmethod
    def type_code(cls) -> str:
        return 'multilabel'

    def get_type_code(self):
        type(self).type_code()


Labeler.valid_type_codes = [
    BinaryLabeler.type_code(),
    MultilabelLabeler.type_code(),
    MulticlassLabeler.type_code()
]
