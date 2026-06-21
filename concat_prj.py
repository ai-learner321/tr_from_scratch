import torch
import torch.nn as nn


class ConcatPrj(nn.Module):
    def __init__(self, d_model=512):
        super(ConcatPrj, self).__init__()
        self.d_model = d_model

        self.prj = nn.Linear(d_model, d_model)  # W_o

    def forward(self, x):
        b, num_heads, seq_len, d_model = x.shape  # attention_output.shape = [B, H, L, Dh]
        concat_x = x.reshape(b, seq_len, -1)
        output = self.prj(concat_x)
        return output


def unit_test():
    x = torch.randn(1, 8, 3, 64)
    b, num_heads, seq_len, d_head = x.shape
    d_model = d_head * num_heads

    concat_prj = ConcatPrj(d_model)
    y = concat_prj(x)
    print(y.shape)


if __name__ == '__main__':
    unit_test()
