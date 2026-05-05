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
        # do the vocab-table lookup
        return self.embedding(x) * math.sqrt(self.d_model)


def unit_test():
    model = TransformerEmbedding(vocab_size=50000, d_model=512)
    print(model)

    x = torch.LongTensor([[7, 42, 105]])  # assume inputs have been tokenizated
    res = model(x)
    print(res.shape)
    print(res)


if __name__ == "__main__":
    unit_test()
