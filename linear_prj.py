import torch
from torch import nn


class LinearPrj(nn.Module):
    '''
    Project the input embedding into qkv vectors
    '''
    def __init__(self, d_in, d_model=512):
        super(LinearPrj, self).__init__()

        self.q = nn.Linear(d_in, d_model)
        self.k = nn.Linear(d_in, d_model)
        self.v = nn.Linear(d_in, d_model)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q, k, v


def unit_test():
    x = torch.randn(1, 3, 512)

    b, seq_len, d_in = x.shape
    linear = LinearPrj(d_in, d_model=512)

    q, k, v = linear(x)
    print(q)
    print(k)
    print(v)


if __name__ == '__main__':
    unit_test()
