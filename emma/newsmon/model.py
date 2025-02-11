import torch
from torch import nn
from torch.functional import F


class RaeExt(nn.Module):

    def __init__(self, values_matrix, index, top_k, k_dim, temperature, dist_metric=False, device='cpu'):
        super(RaeExt, self).__init__()
        # Initialize trainable square matrix to identity
        self.trainable_matrix = nn.Parameter(torch.eye(values_matrix.size(1), device=device))

        # Store other parameters
        self.values_matrix = values_matrix.to(device)
        self.index = index
        self.top_k = top_k
        self.k_dim = k_dim  # num samples
        self.temperature = temperature
        self.dist_metric = dist_metric
        self.device = device


    def forward(self, query_vectors: torch.Tensor):
        """
        Forward pass for the module.

        Args:
            query_vectors (torch.Tensor): Input a batch of query vectors.

        Returns:
            torch.Tensor: Adjusted label probabilities.
        """
        # Perform batched nearest neighbor search using Faiss index
        sim, indices = self.index.search(query_vectors.cpu().numpy(), self.top_k)

        # Convert similarities to PyTorch tensor and move to device
        sim = torch.from_numpy(sim).to(self.device)

        # Apply softmax scaling based on distance metric
        if self.dist_metric:
            sim = F.softmax(-torch.sqrt(sim) / self.temperature, dim=-1)
        else:
            sim = F.softmax(sim / self.temperature, dim=-1)

        # Initialize qKT tensor for the batch with zeros
        batch_size = query_vectors.size(0)
        qkt = torch.zeros((batch_size, self.k_dim), dtype=torch.float32).to(self.device)

        # Assign values to specific indices in qKT tensor
        for i in range(batch_size):
            qkt[i, indices[i]] = sim[i]

        # Compute adjusted probabilities using trainable matrix and values matrix
        weighted_values = torch.matmul(self.values_matrix, self.trainable_matrix)  # Adjust label significance
        probabilities = torch.matmul(qkt, weighted_values)  # Compute final probabilities

        return probabilities
