from typing import List

from FlagEmbedding import BGEM3FlagModel


def load_model():
    return BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)


def embed(model, sentences: List[str]) -> List[List[float]]:
    embeddings = model.encode(
        sentences,
        batch_size=12,
        max_length=8192  # Smaller value will speed up the encoding process.
    )
    return embeddings['dense_vecs']
