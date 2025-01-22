from typing import Callable

import numpy as np
from stanza.models.tokenization.utils import predict


class MLkNN3(object):

    def __init__(self, k, s, knn: Callable[[np.ndarray, int, int], np.ndarray]):
        self.k = k
        self.s = s
        self.label_num = 0
        self.train_data_num = 0
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.Ph1 = np.array([])
        self.Ph0 = np.array(self.label_num)
        self.Peh1 = np.array([self.label_num, self.k + 1])
        self.Peh0 = np.array([self.label_num, self.k + 1])
        self.knn = knn

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        self.train_x = train_x
        self.train_y = train_y
        self.label_num = train_y.shape[1]
        self.train_data_num = train_x.shape[0]

        self.Ph1 = np.zeros(self.label_num)
        self.Ph0 = np.zeros(self.label_num)
        self.Peh1 = np.zeros([self.label_num, self.k + 1])
        self.Peh0 = np.zeros([self.label_num, self.k + 1])

        # computing the prior probabilities
        for i in range(self.label_num):
            cnt = 0
            for j in range(self.train_data_num):
                if train_y[j][i] == 1:
                    cnt = cnt + 1
            self.Ph1[i] = (self.s + cnt) / (self.s * 2 + self.train_data_num)
            self.Ph0[i] = 1 - self.Ph1[i]

        for i in range(self.label_num):

            print('training for label\n', i + 1)
            c1 = np.zeros(self.k + 1)
            c0 = np.zeros(self.k + 1)

            for j in range(self.train_data_num):
                temp = 0
                neighbors = self.knn(train_x, j, self.k)

                for k in range(self.k):
                    temp = temp + int(train_y[int(neighbors[k])][i])

                if train_y[j][i] == 1:
                    c1[temp] = c1[temp] + 1
                else:
                    c0[temp] = c0[temp] + 1

            for j in range(self.k + 1):
                self.Peh1 = (self.s + c1[j]) / (self.s * (self.k + 1) + np.sum(c1))
                self.Peh0 = (self.s + c0[j]) / (self.s * (self.k + 1) + np.sum(c0))


    def predict(self, test_x):
        predict = np.zeros(self.train_y.shape, dtype=np.int64)
        test_data_num = test_x.shape[0]

        for i in range(test_data_num):
            neighbors = self.knn(test_x, i, self.k)

            for j in range(self.label_num):
                temp = 0
                for nei in neighbors:
                    temp = temp + int(self.train_y[int(nei)][j])

                if (self.Ph1[j] * self.Peh1[j][temp] > self.Ph0[j] * self.Peh0[j][temp]):
                    predict[i][j] = 1
                else:
                    predict[i][j] = 0


class MLkNN5:
    def __init__(self, k, s, knn: Callable[[np.ndarray, int], np.ndarray]):
        self.k = k
        self.s = s
        self.label_num = 0
        self.train_data_num = 0
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.Ph1 = np.array([])
        self.Ph0 = np.array([])
        self.Peh1 = np.array([])
        self.Peh0 = np.array([])
        # self.Peh1_t = np.array([])
        # self.Peh0_t = np.array([])
        self.knn = knn

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        # Store training data
        self.train_x = train_x
        self.train_y = train_y
        self.label_num = train_y.shape[1]
        self.train_data_num = train_x.shape[0]

        self.Ph1 = np.zeros((self.label_num,))
        self.Ph0 = np.zeros((self.label_num,))
        self.Peh1 = np.zeros((self.label_num, self.k + 1))
        self.Peh0 = np.zeros((self.label_num, self.k + 1))

        # Initialize probabilities - Compute prior probabilities P(H=1) and P(H=0)
        label_counts = np.sum(self.train_y, axis=0)  # Sum over all samples for each label
        self.Ph1 = (1 + label_counts) / (2 + self.train_data_num)
        self.Ph0 = 1 - self.Ph1

        # Precompute neighbor indices for all training samples
        neighbors_matrix = self.knn(train_x, self.k)  # Shape: (train_data_num, k)

        # Count label occurrences in neighbors
        neighbor_labels = self.train_y[neighbors_matrix]  # Shape: (train_data_num, k, labels_num)
        neighbor_label_counts = np.sum(neighbor_labels, axis=1)  # Shape: (train_data_num, labels_num)

        # Compute conditional probabilities P(E|H=1) and P(E|H=0)
        for label_idx in range(self.label_num):
            label_mask = self.train_y[:, label_idx] == 1

            # Count occurrences of neighbor labels for H=1 and H=0
            c1_counts = np.bincount(neighbor_label_counts[label_mask, label_idx], minlength=self.k + 1)
            c0_counts = np.bincount(neighbor_label_counts[~label_mask, label_idx], minlength=self.k + 1)

            # Smooth and normalize counts to compute probabilities
            c1_sum = c1_counts.sum() + (self.k + 1)
            c0_sum = c0_counts.sum() + (self.k + 1)

            self.Peh1[label_idx] = (c1_counts + 1) / c1_sum
            self.Peh0[label_idx] = (c0_counts + 1) / c0_sum

        # self.Peh1_t = np.zeros((self.label_num, self.k + 1))
        # self.Peh0_t = np.zeros((self.label_num, self.k + 1))
        #
        # for i in range(self.label_num):
        #
        #     print('training for label\n', i + 1)
        #     c1 = np.zeros(self.k + 1)
        #     c0 = np.zeros(self.k + 1)
        #
        #     for j in range(self.train_data_num):
        #         temp = 0
        #         for k in range(self.k):
        #             idx = neighbors_matrix[j][k]
        #             temp = temp + int(train_y[int(idx)][i])
        #
        #         if train_y[j][i] == 1:
        #             c1[temp] = c1[temp] + 1
        #         else:
        #             c0[temp] = c0[temp] + 1
        #
        #     for j in range(self.k + 1):
        #         self.Peh1_t[i][j] = (self.s + c1[j]) / (self.s * (self.k + 1) + np.sum(c1))
        #         self.Peh0_t[i][j] = (self.s + c0[j]) / (self.s * (self.k + 1) + np.sum(c0))
        # peh1_eq = np.array_equal(self.Peh1, self.Peh1_t)
        # peh0_eq = np.array_equal(self.Peh0, self.Peh0_t)
        # print('training for label done')


    def predict(self, test_x: np.ndarray):
        # Precompute neighbor indices for all test samples using multi-vector search
        neighbors_matrix = self.knn(test_x, self.k)

        # Count the number of neighbors with each label for each test sample
        neighbor_labels = self.train_y[neighbors_matrix]   # Shape: (train_data_num, k, labels_num)

        # Sum up label counts for each test sample's neighbors
        neighbor_label_counts = np.sum(neighbor_labels, axis=1)  # Shape: (train_data_num, label_num)

        num_samples = test_x.shape[0]
        predictions = np.zeros((num_samples, self.label_num), dtype=np.int64)

        for i in range(num_samples):
            for j in range(self.label_num):
                temp = 0

                for k in range(self.k):
                    idx = neighbors_matrix[i][k]
                    temp = temp + int(self.train_y[idx][i])

                temp2 = neighbor_label_counts[i][j]
                if temp2 != temp:
                    print('s')

                if (self.Ph1[j] * self.Peh1[j][temp] > self.Ph0[j] * self.Peh0[j][temp]):
                    predictions[i][j] = 1
                else:
                    predictions[i][j] = 0

        # Compute predictions based on probabilities
        predictions1 = np.zeros((num_samples, self.label_num), dtype=np.int64)
        for j in range(self.label_num):  # Iterate over each label
            # Compute probabilities for h=1 and h=0 for all samples
            Ph1_Peh1 = self.Ph1[j] * self.Peh1[j][neighbor_label_counts[:, j]]  # Shape: (num_samples,)
            Ph0_Peh0 = self.Ph0[j] * self.Peh0[j][neighbor_label_counts[:, j]]  # Shape: (num_samples,)

            # Compare probabilities to make predictions
            predictions1[:, j] = (Ph1_Peh1 > Ph0_Peh0).astype(np.int64)

        pred_eq = np.array_equal(predictions, predictions1)
        return predictions