import os
import logging
import tiktoken
import torch
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from .gpt_ds import GPTDatasetV1
from .self_attn import SelfAttentionV1, SelfAttentionV2, MaskedAttentionV1, MaskedAttentionV2, CausalAttention, \
    MultiHeadAttentionWrapper, MultiHeadAttention
from ...core.args import CommonArguments

logger = logging.getLogger('tests.llm')


# noinspection DuplicatedCode
def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_in_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-i', '--data_out_dir'))


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,  # drops the last batch if it is shorter than the specified
                              # batch_size to prevent loss spikes during training
        num_workers=num_workers
    )
    return dataloader


def llm_token(args) -> int:
    """
    Sample that shows manual tokenization with context size 4 and a single target prediction
    """
    with open(os.path.join(args.data_in_dir, 'the-verdict.txt'), 'r', encoding='utf-8') as f:
        raw_text = f.read()
    tokenizer = tiktoken.get_encoding('gpt2')
    enc_text = tokenizer.encode(raw_text)

    enc_sample = enc_text[50:]
    context_size = 4  # A
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size + 1]
    print(f'x: {x}')
    print(f'y:      {y}')

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, '---->', desired)

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), '---->', tokenizer.decode([desired]))

    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)
    data_iter = iter(dataloader)  # python iterator
    first_batch = next(data_iter)
    print(first_batch)

    second_batch = next(data_iter)
    print(second_batch)

    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(f'Inputs:\n {inputs}')
    print(f'\nTargets:\n{targets}')
    return 0


def llm_input_embed(args) -> int:
    input_ids = torch.tensor([2, 3, 5, 1])  # faking some token ids

    vocab_size = 6
    output_dim = 3

    torch.manual_seed(123)
    # the embedding layer is just a more efficient implementation
    # equivalent to the one-hot encoding and matrix-multiplication approach,
    # it can be seen as a neural network layer that can be optimized via backpropagation.
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(f'Embedding layer {embedding_layer.weight}')
    print(f'Embedding at index 3: {embedding_layer(torch.tensor([3]))}')
    print(f'So for tokens (input ids): {input_ids}')
    # it's just a randomly initialized matrix (vocab_size x embedding dim) lookup:
    print(f'Embeddings are:\n{embedding_layer(input_ids)}')
    return 0


def llm_input_embed_pos(args) -> int:
    with open(os.path.join(args.data_in_dir, 'the-verdict.txt'), 'r', encoding='utf-8') as f:
        raw_text = f.read()

    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(f'Token IDs / input ids ({inputs.shape}) :\n\t{inputs}')

    token_embeddings = token_embedding_layer(inputs)
    print(f'Token embeddings has: {token_embeddings.shape}')

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # just a seq 0, 1, 2, ..., context_length - 1
    print(f'Positions: {torch.arange(context_length)}')
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    # the positional embedding tensor consists of four 256-dimensional vectors.
    print(f'Positional embedding layer: {pos_embeddings.shape}')

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
    return 0


def llm_simple_self_attn(args) -> int:
    """
    Beyond viewing the dot product operation as a mathematical tool that combines two vectors to yield a scalar value,
    the dot product is a measure of similarity because it quantifies how much two vectors are aligned:
    a higher dot product indicates a greater degree of alignment or similarity between the vectors.
    In the context of self-attention mechanisms, the dot product determines the extent to which elements in a sequence
    attend to each other: the higher the dot product, the higher the similarity and
    attention score between two elements.
    """
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )

    # remember: dot product is just a sum of element-wise products
    query = inputs[1]  # we consider only second token embedding 'journey'
    attn_scores_2 = torch.empty(inputs.shape[0])  # six dim. vect. to hold attn scores
    for i, x_i in enumerate(inputs):
        # noinspection PyTypeChecker
        attn_scores_2[i] = torch.dot(x_i, query)

    print(f'Inputs:\n {inputs}\n')
    print(f'Attention scores@2:\n{attn_scores_2}\n')

    # This normalization is a convention that is useful for interpretation and for maintaining
    # training stability in an LLM.
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print(f'Attention scores@2 normalized = Attention weights@2  (mean norm. is used here):\n{attn_weights_2_tmp}\n')
    print(f'Sum: {attn_weights_2_tmp.sum()}\n')

    # In practice, it's more common and advisable to use the softmax function for normalization.
    # This approach is better at managing extreme values and offers more favorable gradient properties during training.
    def softmax_naive(x):
        return torch.exp(x) / torch.exp(x).sum(dim=0)

    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print(f'Attention weights@2 (softmax norm. is used here):\n{attn_weights_2_naive}')
    print(f'Sum: {attn_weights_2_naive.sum()}\n')

    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print(f'\nAttention weights@2 (torch softmax norm. is used here):\n{attn_weights_2}')
    print(f'Sum: {attn_weights_2.sum()}\n')

    # The final step is to compute the context vector
    # This context vector is a combination of all input vectors x^1 to x^6 weighted by the attention weights.
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i  # weighted sum
    print(f'Context vector@2: {context_vec_2}\n')

    # Generalization to all inputs
    # To generalize to all input not just to the one at index 2
    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            # noinspection PyTypeChecker
            attn_scores[i, j] = torch.dot(x_i, x_j)
    print(f'Attention scores (hand computation):\n{attn_scores}\n')
    attn_scores = inputs @ inputs.T
    print(f'Attention scores (using matrix multiplication):\n{attn_scores}\n')
    attn_weights = torch.softmax(attn_scores, dim=-1)  # -1 => apply the normalization along the last dimension
    print(f'Attention weights (torch softmax):\n{attn_weights}\n')

    context_vecs = attn_weights @ inputs
    print(f'Context vectors: \n{context_vecs}\n')

    return 0


# noinspection PyPep8Naming
def llm_train_self_attn(args) -> int:
    """
    we want to compute context vectors as weighted sums over the input vectors specific to a certain input element.
    There are only slight differences compared to the basic self-attention mechanism above.
    The most notable difference is the introduction of weight matrices that are updated during model training.
    These trainable weight matrices are crucial so that the model (specifically, the attention module inside the model)
    can learn to produce 'good' context vectors.
    """
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    x_2 = inputs[1]
    d_in = inputs.shape[1]  # Note that in GPT-like models, the input and output dimensions are usually the same.
    d_out = 2               # to better follow the computation, we choose different input (d_in=3) and output (d_out=2)

    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # trainable weight matrices
    W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # grad=False => to reduce clutter
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    # This is not to be confused with the attention weights. Attention weights determine the extent to which
    # a context vector depends on the different parts of the input - to what extent the network focuses
    # on different parts of the input.
    query_2 = x_2 @ W_query  # A "query" is analogous to a search query in a database.
    key_2   = x_2 @ W_key    # The "key" is like a database key used for indexing and searching.
    value_2 = x_2 @ W_value  # The "value" in this context is similar to the value in a key-value pair in a database.
    print(f'Query@2: {query_2}\n')

    keys   = inputs @ W_key
    values = inputs @ W_value
    print(f'Inputs {inputs.shape} projected to Keys {keys.shape} and values {values.shape} via weight matrices.')
    # we projected the 6 input tokens from a 3D onto a 2D embedding space (torch.Size([6, 2]))

    attn_scores_2 = query_2 @ keys.T  # All attention scores for given query
    d_k = keys.shape[-1]
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)  # The difference to earlier is that we now
    # scale the attention scores by dividing them by the square root of the embedding dimension of the keys
    print(f'Attention weights@2 (torch softmax):\n{attn_weights_2}\n')
    context_vec_2 = attn_weights_2 @ values
    print(f'Context vector@2: {context_vec_2}\n')

    torch.manual_seed(123)
    sa = SelfAttentionV1(d_in, d_out)
    print(f'Self attention v1 (simple impl): \n{sa(inputs)}\n')
    sa = SelfAttentionV2(d_in, d_out)
    print(f'Self attention v2 (Linear Layers): \n{sa(inputs)}\n')
    sa = MaskedAttentionV1(d_in, d_out)
    print(f'Self attention v3 (Linear Layers + mask + mean norm.): \n{sa(inputs)}\n')

    torch.manual_seed(123)
    sa = MaskedAttentionV2(d_in, d_out)
    print(f'Self attention v4 (Linear Layers + mask + softmax norm.): \n{sa(inputs)}\n')

    torch.manual_seed(123)
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]
    sa = CausalAttention(d_in, d_out, context_length, 0.0)
    print(f'Self attention v5 (Linear Layers + mask + softmax norm. + dropout - optimized): \n{sa(batch)}\n')

    torch.manual_seed(123)
    sa = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0, num_heads=2)
    print(f'Multi-head attention wrapper v6: \n{sa(batch)}\n')

    torch.manual_seed(123)
    sa = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=2)
    print(f'Multi-head attention v7: \n{sa(batch)}\n')

    return 0
