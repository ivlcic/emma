import torch
import torch.nn as nn

from torch import Tensor


class SelfAttentionV1(nn.Module):

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # The terms "key," "query," and "value" in the context of attention mechanisms are borrowed from the domain of
        # information retrieval and databases, where similar concepts are used to store, search, and
        # retrieve information.
        self.d_out = d_out
        # A "query" is analogous to a search query in a database. It represents the current item
        # (e.g., a word or token in a sentence) the model focuses on or tries to understand.
        # The query is used to probe the other parts of the input sequence to determine
        # how much attention to pay to them.
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        # The "key" is like a database key used for indexing and searching. In the attention mechanism,
        # each item in the input sequence (e.g., each word in a sentence) has an associated key.
        # These keys are used to match with the query.
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        # The "value" in this context is similar to the value in a key-value pair in a database.
        # It represents the actual content or representation of the input items.
        # Once the model determines which keys (and thus which parts of the input) are most relevant
        # to the query (the current focus item), it retrieves the corresponding values.
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x: Tensor) -> Tensor:
        # The reason for the normalization by the embedding dimension size is to improve the training performance
        # by avoiding small gradients. For instance, when scaling up the embedding dimension,
        # which is typically greater than a thousand for GPT-like LLMs, large dot products can result in very
        # small gradients during backpropagation due to the softmax function applied to them.
        # As dot products increase, the softmax function behaves more like a step function,
        # resulting in gradients nearing zero.
        # These small gradients can drastically slow down learning or cause training to stagnate.
        # The scaling by the square root of the embedding dimension is the reason why this self-attention mechanism
        # is also called scaled-dot product attention.
        keys    = x @ self.W_key
        queries = x @ self.W_query
        values  = x @ self.W_value
        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class SelfAttentionV2(nn.Module):

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: Tensor) -> Tensor:
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class MaskedAttentionV1(SelfAttentionV2):
    """
    Causal attention, also known as masked attention, is a specialized form of self-attention.
    It restricts a model to only consider previous and current inputs in a sequence when processing any given token.
    This is in contrast to the standard self-attention mechanism, which allows access to the entire input sequence
    at once.
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__(d_in, d_out, qkv_bias)

    def forward(self, x: Tensor) -> Tensor:
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        attn_scores = queries @ keys.T  # omega
        context_length = attn_scores.shape[0]
        mask_simple = torch.tril(torch.ones(context_length, context_length))  # lower triangular
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        masked_simple = attn_weights * mask_simple         # consider only lower triangular values + diagonal
        row_sums = masked_simple.sum(dim=1, keepdim=True)  # Re-normalize (mean normalization)
        masked_simple_norm = masked_simple / row_sums      # Re-normalize (mean normalization)

        context_vec = masked_simple_norm @ values
        return context_vec


class MaskedAttentionV2(SelfAttentionV2):
    """
    Causal attention, also known as masked attention, is a specialized form of self-attention.
    Implemented using softmax.
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__(d_in, d_out, qkv_bias)

    def forward(self, x: Tensor) -> Tensor:
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        attn_scores = queries @ keys.T  # omega

        context_length = attn_scores.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

        # The softmax function converts its inputs into a probability distribution.
        # When negative infinity values (-∞) are present in a row, the softmax function treats them as zero
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)

        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    """
    Causal attention, also known as masked attention, implemented using a softmax normalization,
    supporting batch input and dropout.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float = 0.5, qkv_bias: bool = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, d_in = x.shape  # batch size, num tokens in a sequence, input dimension

        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)  # First dimension is batch so we transpose second and third

        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # in pytorch "_" function postfix is in-place operation w/o memory copy overhead
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)]
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Use a Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)    # Tensor shape: (b, num_tokens, d_out)
        queries = self.W_query(x)  # Tensor shape: (b, num_tokens, d_out)
        values  = self.W_value(x)  # Tensor shape: (b, num_tokens, d_out)

        # We implicitly split the matrix by adding a `num_heads` dimension.
        # Then we unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys    = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values  = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # Compute dot product for each head
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # Mask truncated to the number of tokens
        attn_scores.masked_fill_(mask_bool, -torch.inf)  # Use the mask to fill attention scores
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)  # Tensor shape: (b, num_tokens, n_heads, head_dim)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # Add an optional linear projection
        return context_vec
