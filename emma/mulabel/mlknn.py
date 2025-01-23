from typing import Callable, Tuple

import numpy as np


class MLkNN:
    def __init__(self, model_name: str, k, s, knn: Callable[[str, np.ndarray, int], np.ndarray]):
        self.model_name = model_name
        self.k = k
        self.s = s
        self.n_label = 0
        self.n_train = 0
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.ph1 = np.array([])
        self.ph0 = np.array([])
        self.peh1 = np.array([])
        self.peh0 = np.array([])
        self.knn = knn

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        # Store training data
        self.train_x = train_x  # Shape: (n_train, x_dim)
        self.train_y = train_y  # Shape: (n_train, n_label)
        self.n_label = train_y.shape[1]
        self.n_train = train_x.shape[0]

        self.ph1 = np.zeros((self.n_label,))
        self.ph0 = np.zeros((self.n_label,))
        self.peh1 = np.zeros((self.n_label, self.k + 1))
        self.peh0 = np.zeros((self.n_label, self.k + 1))

        # Initialize probabilities - Compute prior probabilities P(H=1) and P(H=0)
        label_counts = np.sum(self.train_y, axis=0)  # Sum over all samples for each label
        self.ph1 = (self.s + label_counts) / (self.s * 2 + self.n_train)  # Shape (n_label)
        self.ph0 = 1.0 - self.ph1  # Shape (n_label)

        # Precompute neighbor indices for all training samples
        neighbors_matrix = self.knn(self.model_name, train_x, self.k)  # Shape: (n_train, k)

        # Count label occurrences in neighbors
        neighbor_labels = self.train_y[neighbors_matrix]  # Shape: (n_train, k, n_label)
        neighbor_label_counts = np.sum(neighbor_labels, axis=1)  # Shape: (n_train, n_label)

        # Compute conditional probabilities P(E|H=1) and P(E|H=0)
        for l_idx in range(self.n_label):
            label_mask = self.train_y[:, l_idx] == 1

            # Count occurrences of neighbor labels for H=1 and H=0
            c1_counts = np.bincount(neighbor_label_counts[label_mask, l_idx], minlength=self.k + 1)  # Shape (51)
            c0_counts = np.bincount(neighbor_label_counts[~label_mask, l_idx], minlength=self.k + 1)  # Shape (51)

            # Smooth and normalize counts to compute probabilities
            c1_sum = self.s * (self.k + 1) + c1_counts.sum()
            c0_sum = self.s * (self.k + 1) + c0_counts.sum()

            self.peh1[l_idx] = (self.s + c1_counts) / c1_sum  # Shape (n_label, k)
            self.peh0[l_idx] = (self.s + c0_counts) / c0_sum  # Shape (n_label, k)


    def predict(self, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Precompute neighbor indices for all test samples using multi-vector search
        neighbors_matrix = self.knn(self.model_name, test_x, self.k)

        # Count the number of neighbors with each label for each test sample
        neighbor_labels = self.train_y[neighbors_matrix]   # Shape: (n_test, k, n_label)

        # Sum up label counts for each test sample's neighbors
        neighbor_label_counts = np.sum(neighbor_labels, axis=1)  # Shape: (n_test, n_label)

        num_samples = test_x.shape[0]

        # Compute predictions based on probabilities
        predictions = np.zeros((num_samples, self.n_label), dtype=np.int64)
        probabilities = np.zeros((num_samples, self.n_label), dtype=np.float32)
        for l_idx in range(self.n_label):  # Iterate over each label
            # Compute probabilities for h=1 and h=0 for all samples
            ph1_peh1 = self.ph1[l_idx] * self.peh1[l_idx][neighbor_label_counts[:, l_idx]]  # Shape: (n_test,)
            ph0_peh0 = self.ph0[l_idx] * self.peh0[l_idx][neighbor_label_counts[:, l_idx]]  # Shape: (n_test,)
            probabilities[:, l_idx] = ph1_peh1 / (ph0_peh0 + ph1_peh1)
            # Compare probabilities to make predictions
            predictions[:, l_idx] = (ph1_peh1 > ph0_peh0).astype(np.int64)

        return predictions, probabilities


class MLkNNAlt:
    def __init__(self, model_name: str, k, knn: Callable[[str, np.ndarray, int], np.ndarray], smooth=1.0, threshold=0.5):
        self.k = k
        self.s = smooth
        self.prior = None
        self.posterior = None
        self.threshold = threshold
        self.model_name = model_name
        self.knn = knn
        self.n_labels = 0

    def __set_params(self, X, y):
        n_labels = y.shape[1]
        self.__dict__.update({'n_labels': n_labels})


    def __get_nn_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.knn(self.model_name, X, self.k)

    def __set_prior(self, y_train):
        prior = np.zeros((self.n_labels, 2))
        prior[:, 1] = (self.s + y_train.sum(axis=0)) / (self.s * 2 + y_train.shape[0])
        prior[:, 0] = 1 - prior[:, 1]
        setattr(self, 'prior', prior)

    def __set_posterior(self, C_x, y):
        posterior = np.zeros((self.n_labels, self.k + 1, 2))
        for label in range(self.n_labels):
            c = np.zeros(self.k + 1)
            c_p = np.zeros(self.k + 1)
            for row in range(C_x.shape[0]):
                d = C_x[row, label]
                if y[row, label] == 1:
                    c[d] += 1
                else:
                    c_p[d] += 1
            for neighbor in range(self.k + 1):
                posterior[label, neighbor, 1] = (self.s + c[neighbor]) / (self.s * (self.k + 1) + c.sum())
                posterior[label, neighbor, 0] = (self.s + c_p[neighbor]) / (self.s * (self.k + 1) + c_p.sum())

        setattr(self, 'posterior', posterior)

    def __get_membership_counting_vectors(self, nn_mat, y_train):
        return np.array([y_train[row].sum(axis=0) for row in nn_mat], dtype=int)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        if train_y.shape[1] < 2:
            raise ValueError("Target must be one-hot encoded label vectors")

        self.__dict__.update({'__x_train': train_x, '__y_train': train_y})

        self.__set_params(train_x, train_y)
        self.__set_prior(train_y)
        nn_mat = self.__get_nn_matrix(train_x)

        counting_x = self.__get_membership_counting_vectors(nn_mat, train_y)
        self.__set_posterior(counting_x, train_y)

    def predict(self, text_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = np.zeros((text_x.shape[0], self.n_labels))
        r_pred = np.zeros((text_x.shape[0], self.n_labels))
        nn_mat = self.__get_nn_matrix(text_x)
        C_x = self.__get_membership_counting_vectors(nn_mat, getattr(self, '__y_train'))
        for label in range(self.n_labels):
            for row in range(C_x.shape[0]):
                y_t_1 = self.prior[label, 1] * self.posterior[label, C_x[row, label], 1]
                y_t_0 = self.prior[label, 0] * self.posterior[label, C_x[row, label], 0]
                r_pred[row, label] = y_t_1 / (y_t_0 + y_t_1)
                y_pred[row, label] = int(y_t_0 <= y_t_1)

        return y_pred.astype(int),  r_pred
