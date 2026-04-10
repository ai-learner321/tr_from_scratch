import torch
import torch.nn as nn


class ConcatPrj(nn.Module):
    def __init__(self, d_model=512):
        super(ConcatPrj, self).__init__()
        self.d_model = d_model

        self.prj = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, seq_len, num_heads, d_model = x.shape
        concat_x = x.reshape(b, seq_len, -1)
        y = self.prj(concat_x)
        return y


def unit_test():
    x = torch.randn(1, 3, 8, 64)
    b, seq_len, num_heads, d_head = x.shape
    d_model = d_head * num_heads

    concat_prj = ConcatPrj(d_model)
    y = concat_prj(x)
    print(y.shape)


if __name__ == '__main__':
    unit_test()
