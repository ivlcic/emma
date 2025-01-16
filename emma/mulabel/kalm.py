import torch
from sentence_transformers import SentenceTransformer
from transformers.utils import is_torch_npu_available


class KaLMEmbedding(torch.nn.Module):
    def __init__(self,
                 model_name: str = None,
                 normalized: bool = True,
                 use_fp16: bool = True,
                 device: str = None
                ):
        super().__init__()
        self.normalized = normalized
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif is_torch_npu_available():
                self.device = torch.device("npu")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False
        self.use_fp16 = use_fp16
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self,
               texts: None,
               batch_size: int = 16):
        if isinstance(texts, str):
            texts = [texts]
        num_texts = len(texts)

        all_dense_vecs = []
        for n, i in enumerate(range(0, num_texts, batch_size)):
            batch = texts[i: i + batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                convert_to_numpy=False
            )
            all_dense_vecs.append(embeddings)

        all_dense_vecs = torch.cat(all_dense_vecs, dim=0).cpu().numpy()
        return {
            "dense_embeddings": all_dense_vecs
        }
