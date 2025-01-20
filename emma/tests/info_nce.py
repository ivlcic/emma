import torch
import logging
import torch.nn.functional as F


from argparse import ArgumentParser

from ..core.args import CommonArguments

logger = logging.getLogger('tests.ir_metrics')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_in_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-i', '--data_out_dir'))


def info_nce_test(args):
    # Batch size and embedding dimensions
    batch_size = 16
    embedding_dim = 128
    num_docs_per_query = 4  # Number of documents (positive + 3 negatives) per query
    temperature = 0.1

    # Simulated dense vectors for queries and documents
    q_dense_vecs = torch.randn(batch_size, embedding_dim)  # Query embeddings: (B, D)
    p_dense_vecs = torch.randn(batch_size * num_docs_per_query, embedding_dim)  # Document embeddings: (B * N, D)

    # Create targets
    idxs = torch.arange(q_dense_vecs.size(0), device=q_dense_vecs.device, dtype=torch.long)
    targets = idxs * (p_dense_vecs.size(0) // q_dense_vecs.size(0))  # Targets point to positive docs

    # Define a dense scoring function (e.g., dot product similarity)
    def dense_score(queries, docs):
        return torch.matmul(queries, docs.T)  # Compute similarity scores: (B, B * N)

    # Compute dense scores
    dense_scores = dense_score(q_dense_vecs, p_dense_vecs)  # Shape: (B, B * N)

    # Compute loss
    loss = F.cross_entropy(dense_scores / temperature, targets, reduction='mean')
    print(f"Cross-Entropy Loss: {loss.item()}")
    return 0


# non-working example just a thought
def _info_nce_test2(
        self,
        model,
        inputs,
        **kwargs):
    query = inputs["query"]
    pos = inputs["pos"]
    neg = inputs["neg"]

    text_embeddings = model(query, max_len=self.args.query_max_len)

    text_pos_embeddings = model(
        pos,
        max_len=self.args.passage_max_len,
    )
    text_neg_embeddings = model(
        neg,
        max_len=self.args.passage_max_len,
    )

    sim_pos_vector = torch.cosine_similarity(
        text_embeddings, text_pos_embeddings, dim=-1
    )
    sim_pos_vector = sim_pos_vector / self.args.temperature
    sim_neg_matrix = torch.cosine_similarity(
        text_embeddings.unsqueeze(1),
        text_neg_embeddings.unsqueeze(0),
        dim=-1,
    )
    sim_neg_matrix = sim_neg_matrix / self.args.temperature
    sim_diff_matrix = sim_pos_vector.unsqueeze(1) - sim_neg_matrix
    loss = -torch.log(torch.sigmoid(sim_diff_matrix)).mean()
    return loss
