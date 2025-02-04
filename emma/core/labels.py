import logging

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union

from numpy import ndarray
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, LabelEncoder

logger = logging.getLogger('core.labels')


class Labeler(ABC):

    def __init__(self, encoder, labels: Optional[List] = None):
        self.encoder = encoder
        self.labels = labels if labels else []
        self.computed = False
        self.num_labels = len(self.labels)

    def collect(self, labels: List[Any]):
        for x in labels:
            if isinstance(x, (list, set, tuple)):
                [self.labels.append(a) for a in x if a not in self.labels]
            elif x not in self.labels:
                self.labels.append(x)

    def fit(self):
        self.computed = True
        self.num_labels = len(self.labels)

    @abstractmethod
    def get_type_code(self):
        pass

    def vectorize(self, labels: Union[List, ndarray]) -> ndarray:
        if not self.computed:
            self.fit()

        result = self.encoder.transform(labels)
        return result

    def unvectorize(self, vector: ndarray) -> List:
        if not self.computed:
            self.fit()

        result = self.encoder.inverse_transform(vector)
        return result

    def labels_to_ids(self):
        if not self.computed:
            self.fit()
        label_to_ids = dict(zip(self.encoder.classes_, range(len(self.encoder.classes_))))
        return label_to_ids

    def ids_to_labels(self):
        if not self.computed:
            self.fit()
        ids_to_labels = {index: label for index, label in enumerate(self.encoder.classes_)}
        return ids_to_labels


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

    def unvectorize(self, vector: ndarray) -> List:
        inverse_result = super().unvectorize(vector)

        if len(inverse_result) == 1:  # Check if there's only one tuple in the list the return single list
            return list(inverse_result[0])
        return inverse_result


Labeler.valid_type_codes = [
    BinaryLabeler().get_type_code(),
    MultilabelLabeler().get_type_code(),
    MulticlassLabeler().get_type_code()
]
