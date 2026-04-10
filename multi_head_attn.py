import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tr_from_scratch.linear_prj import LinearPrj
from tr_from_scratch.split_into_heads import SplitIntoHeads
from tr_from_scratch.concat_prj import ConcatPrj


class MultiHeadAttn(nn.Module):
    '''
    Vanilla Multi-Head Attention
    '''
    def __init__(self, n_heads=8, d_model=512):
        super(MultiHeadAttn, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        self.linear_prj = LinearPrj(d_model)
        self.split_into_heads = SplitIntoHeads(n_heads=n_heads)
        self.concat_prj = ConcatPrj(d_model)

    def forward(self, x):
        q, k, v = self.linear_prj(x)
        splited_q = self.split_into_heads(q)
        splited_k = self.split_into_heads(k)
        splited_v = self.split_into_heads(v)

        dot_product = torch.matmul(splited_q, splited_k.transpose(2, 3))
        scaled_dot_product = dot_product / math.sqrt(self.d_model)
        scores = F.softmax(scaled_dot_product, dim=-1)
        attn = torch.matmul(scores, splited_v)
        y = self.concat_prj(attn)
        return y


def unit_test():
    x = torch.randn(1, 3, 512)
    b, seq_len, d_model = x.shape

    mha = MultiHeadAttn(n_heads=8, d_model=d_model)
    y = mha(x)
    print(y.shape)
    print(y)


if __name__ == '__main__':
    unit_test()
