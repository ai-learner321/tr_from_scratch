import math
import torch
import torch.nn as nn


class TransformerEmbedding(nn.Module):
    '''
    Take a lookup from the vocabulary table to find the embedding vector.
    '''
    def __init__(self, vocab_size=50000, d_model=512):
        super(TransformerEmbedding, self).__init__()

        self.d_model = d_model  # hidden dim of model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # return self.embedding(x) * math.sqrt(self.d_model)

        found = self.embedding(x)  # do the vocab-table lookup
        norm_found = found * math.sqrt(self.d_model)
        return norm_found


def test_embedding_output_shape():
    vocab_size = 50000
    d_model = 512
    tr_embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)

    b, seq_len = 2, 5
    x = torch.randint(0, vocab_size, (b, seq_len), dtype=torch.long)
    output = tr_embedding(x)

    assert output.shape == (b, seq_len, d_model)


def test_embedding_with_known_weight():
    vocab_size = 4
    d_model = 2
    tr_embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)

    known_weight = torch.tensor([
        [1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.],
    ])

    with torch.no_grad():
        tr_embedding.embedding.weight.copy_(known_weight)

    x = torch.tensor([
        [0, 2],
        [1, 3],
    ])

    output = tr_embedding(x)

    expected = torch.tensor([
        [
            [1., 2.],
            [5., 6.],
        ],
        [
            [3., 4.],
            [7., 8.],
        ]
    ]) * math.sqrt(d_model)

    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-5)


def test_same_token_has_same_embedding():
    torch.manual_seed(0)
    vocab_size = 100
    d_model = 16
    tr_embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)

    x = torch.tensor([
        [7, 1, 7],
        [2, 3, 7]
    ], dtype=torch.long)

    output = tr_embedding(x)

    output_00 = output[0, 0]
    output_02 = output[0, 2]

    assert torch.allclose(output[0, 0], output[0, 2])
    assert torch.allclose(output[0, 0], output[1, 2])


def test_embedding_backward():
    torch.manual_seed(0)
    vocab_size = 100
    d_model = 16
    tr_embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)
    x = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ], dtype=torch.long)
    output = tr_embedding(x)

    loss = output.mean()
    loss.backward()

    grad = tr_embedding.embedding.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_only_used_tokens_receive_gradients():
    torch.manual_seed(0)
    vocab_size = 10
    d_model = 16
    tr_embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)

    x = torch.tensor([
        [1, 3, 5],
        [5, 1, 5]
    ], dtype=torch.long)
    output = tr_embedding(x)

    loss = output.sum()
    loss.backward()

    grad = tr_embedding.embedding.weight.grad

    used_token_ids = [1, 3, 5]
    for token_id in range(vocab_size):
        if token_id in used_token_ids:
            assert not torch.allclose(grad[token_id],
                                  torch.zeros_like(grad[token_id]))
        else:
            assert torch.allclose(grad[token_id], torch.zeros_like(grad[token_id]))


def main():
    # test_embedding_output_shape()
    # test_embedding_with_known_weight()
    # test_same_token_has_same_embedding()
    # test_embedding_backward()
    test_only_used_tokens_receive_gradients()
    pass


if __name__ == "__main__":
    main()
