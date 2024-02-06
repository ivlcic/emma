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
        self.num_labels = len(self.labels)

    def collect(self, labels: List[Any]):
        for x in labels:
            if isinstance(x, (list, set, tuple)):
                self.labels.update(x)
            else:
                self.labels.add(x)

    def fit(self):
        self.computed = True
        self.num_labels = len(self.labels)

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
        if self.num_labels != 2:
            raise ValueError('Invalid data was passed into Labeler collect. Must have at least two values for label!')
        self.num_labels = 1  # for binary classification we have a single label with two values
        logger.debug('Total number of labels: %s', self.num_labels)

    def get_type_code(self):
        return 'binary'


class MulticlassLabeler(Labeler):

    def __init__(self, labels: Optional[List] = None):
        super().__init__(LabelEncoder(), labels)

    def fit(self):
        self.encoder.fit(list(self.labels))
        super().fit()
        logger.debug('Total number of labels: %s', self.num_labels)

    def get_type_code(self):
        return 'multiclass'


class MultilabelLabeler(Labeler):

    def __init__(self, labels: Optional[List] = None):
        super().__init__(MultiLabelBinarizer(), labels)

    def fit(self):
        self.encoder.fit([list(self.labels)])
        super().fit()
        logger.debug('Total number of labels: %s', self.num_labels)

    def get_type_code(self):
        return 'multilabel'


Labeler.valid_type_codes = [
    BinaryLabeler().get_type_code(),
    MultilabelLabeler().get_type_code(),
    MulticlassLabeler().get_type_code()
]
