#@Time      :2018/9/14 14:27
#@Author    :zhounan
# @FileName: mlknn.py
from typing import Callable

import numpy as np
from numpy import ndarray


class MLKNN:

    def __init__(self, train_x: ndarray, train_y: ndarray, k: int, s: int, knn: Callable[[ndarray, int, int], ndarray]):
        self.k = k
        self.s = s
        self.train_x = train_x
        self.train_y = train_y
        self.label_num = train_y.shape[1]
        self.train_data_num = train_x.shape[0]
        self.Ph1 = np.zeros(self.label_num)
        self.Ph0 = np.zeros(self.label_num)
        self.Peh1 = np.zeros([self.label_num, self.k + 1])
        self.Peh0 = np.zeros([self.label_num, self.k + 1])
        self.knn = knn

    def train(self):
        #computing the prior probabilities
        # for i in range(self.label_num):
        #     cnt = 0
        #     for j in range(self.train_data_num):
        #         if self.train_y[j][i] == 1:
        #             cnt = cnt + 1
        #     self.Ph1[i] = (self.s + cnt) / (self.s * 2 + self.train_data_num)
        #     self.Ph0[i] = 1 - self.Ph1[i]

        # Optimized computation of prior probabilities
        cnt = np.sum(self.train_y == 1, axis=0)  # Count occurrences of 1 for each label
        self.Ph1 = (self.s + cnt) / (self.s * 2 + self.train_data_num)  # Calculate P(h=1)
        self.Ph0 = 1 - self.Ph1  # Calculate P(h=0)

        for i in range(self.label_num):

            print('training for label\n', i + 1)
            c1 = np.zeros(self.k + 1)
            c0 = np.zeros(self.k + 1)

            for j in range(self.train_data_num):
                temp = 0
                neighbors = self.knn(self.train_x, j, self.k)

                for k in range(self.k):
                    temp = temp + int(self.train_y[int(neighbors[k])][i])

                if self.train_y[j][i] == 1:
                    c1[temp] = c1[temp] + 1
                else:
                    c0[temp] = c0[temp] + 1

            for j in range(self.k + 1):
                self.Peh1 = (self.s + c1[j]) / (self.s * (self.k + 1) + np.sum(c1))
                self.Peh0 = (self.s + c0[j]) / (self.s * (self.k + 1) + np.sum(c0))


class MLKNN2:
    def __init__(self, k: int, s: int, knn: Callable[[ndarray, int, int], ndarray]):
        self.s = s
        self.k = k
        self.Ph1 = np.array([])
        self.Ph0 = np.array([])
        self.Peh1 = np.array([])
        self.Peh0 = np.array([])
        self.knn = knn

    def fit(self, train_x: ndarray, train_y: ndarray):
        self.labels_num = train_y.shape[1]
        self.train_data_num = train_x.shape[0]

        # Vectorized computation of Ph1 and Ph0
        label_counts = np.sum(train_y, axis=0)  # Count occurrences of 1 for each label
        self.Ph1 = (self.s + label_counts) / (self.s * 2 + self.train_data_num)
        self.Ph0 = 1 - self.Ph1

        # Precompute neighbors for all training samples
        neighbors = [self.knn(train_x, j, self.k) for j in range(self.train_data_num)]
        all_neighbors = np.array(neighbors)

        # Vectorized computation of Peh1 and Peh0
        for i in range(self.labels_num):
            # Compute the number of neighbors with label=1 for each training sample
            temp = np.sum(train_y[all_neighbors, i], axis=1)

            # Count occurrences of each value in temp for c1 and c0
            c1 = np.bincount(temp[train_y[:, i] == 1], minlength=self.k + 1)
            c0 = np.bincount(temp[train_y[:, i] == 0], minlength=self.k + 1)

            # Compute probabilities Peh1 and Peh0
            total_c1 = np.sum(c1)
            total_c0 = np.sum(c0)
            self.Peh1[i] = (self.s + c1) / (self.s * (self.k + 1) + total_c1)
            self.Peh0[i] = (self.s + c0) / (self.s * (self.k + 1) + total_c0)

    def predict(self, _test_data):
        test_data_num = _test_data.shape[0]
        rtl = np.zeros((test_data_num, self.labels_num))
        predict_labels = np.zeros((test_data_num, self.labels_num))

        # Precompute neighbors for all test samples
        all_test_neighbors = np.array([self.knn(self.train_data, _test_data[i], self.k) for i in range(test_data_num)])

        # Vectorized prediction computation
        for j in range(self.labels_num):
            # Compute the number of neighbors with label=1 for each test sample
            temp = np.sum(self.train_target[all_test_neighbors, j], axis=1)

            # Compute probabilities y1 and y0 for each test sample
            y1 = self.Ph1[j] * self.Peh1[j][temp]
            y0 = self.Ph0[j] * self.Peh0[j][temp]

            # Compute rtl and pred
            # Compute rtl and predict labels based on y1 and y0
            rtl[:, j] = y1 / (y1 + y0)
            predict_labels[:, j] = (y1 > y0).astype(int)

        return predict_labels