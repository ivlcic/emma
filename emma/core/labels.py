import logging
from typing import List, Optional, Any

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

logger = logging.getLogger('core.labels')


class Labeler:

    def __init__(self, binarizer, labels: Optional[List] = None):
        self.binarizer = binarizer
        if labels is not None:
            self.all_labels = labels
        else:
            self.all_labels = []
        self.computed = False
        self.num_labels = 0

    def extend(self, labels: List[Any]):
        self.all_labels.extend(labels)

    def fit(self):
        self.num_labels = len(self.binarizer.classes_)
        self.computed = True
        logger.debug('Total number of labels: %s', self.num_labels)

    def vectorize(self, labels: List) -> List:
        if not self.computed:
            self.fit()

        result = self.binarizer.transform(labels)
        return result


class MultiLabeler(Labeler):
    def __init__(self, labels: Optional[List] = None):
        super().__init__(MultiLabelBinarizer(), labels)

    def fit(self):
        self.binarizer.fit([self.all_labels])
        super().fit()


class BinaryLabeler(Labeler):
    def __init__(self, labels: Optional[List] = None):
        super().__init__(LabelBinarizer(), labels)
        self.num_labels = 2
        self.computed = True

    def fit(self):
        self.binarizer.fit(self.all_labels)
        super().fit()
