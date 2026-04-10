import torch
import torch.nn as nn


class SplitIntoHeads(nn.Module):
    '''
    Split the qkv vectors into multiple heads and transpose them
    for matrix multiplication.
    '''
    def __init__(self, n_heads=8):
        super(SplitIntoHeads, self).__init__()
        self.n_heads = n_heads

    def forward(self, x):
        b, seq_len, d_model = x.shape
        d_head = d_model // self.n_heads
        splited_x = x.reshape(b, seq_len, self.n_heads, d_head)
        y = splited_x.transpose(1, 2)
        return y


def unit_test():
    x = torch.randn(1, 3, 512)
    spliter = SplitIntoHeads(n_heads=8)
    y = spliter(x)
    print(y.shape)


if __name__ == "__main__":
    unit_test()
