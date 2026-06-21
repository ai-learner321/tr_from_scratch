import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tr_from_scratch.linear_prj import LinearPrj
from tr_from_scratch.split_into_heads import SplitIntoHeads
from tr_from_scratch.concat_prj import ConcatPrj


class MultiHeadSelfAttn(nn.Module):
    '''
    Vanilla Multi-Head Self Attention.

    input:
        x: (batch_size, seq_len, d_model)
        padding_mask: (batch_size, seq_len), true represents valid token, false represents
            padding token
        casual mask: forbid the current position paying attention to the future position
    '''
    def __init__(self,
                 n_heads=8,
                 d_model=512,
                 attn_dropout=0.0,
                 proj_dropout=0.0):
        super(MultiHeadSelfAttn, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

        self.linear_prj = LinearPrj(d_model)
        self.split_into_heads = SplitIntoHeads(n_heads=n_heads)
        self.concat_prj = ConcatPrj(d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x,
                padding_mask=None,
                causal_mask=False):
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model

        q, k, v = self.linear_prj(x)
        # x.shape = [B, L, D], B：batch size; L：序列长度; D：d_model
        # q.shape = [B, L, D]
        # k.shape = [B, L, D]
        # v.shape = [B, L, D]

        splited_q = self.split_into_heads(q)
        splited_k = self.split_into_heads(k)
        splited_v = self.split_into_heads(v)
        # q.shape = [B, H, L, Dh], Dh = D // H
        # k.shape = [B, H, L, Dh]
        # v.shape = [B, H, L, Dh]

        attn_logits = torch.matmul(splited_q, splited_k.transpose(2, 3))
        # attn_logits.shape = [B, H, L, L]

        scaled_attn_logits = attn_logits / math.sqrt(self.d_head)
        # scaled_attn_logis.shape = [B, H, L, L]

        if padding_mask is not None:
            key_mask = padding_mask[:, None, None, :]  # [B, L] -> [B, 1, 1, L], broadcastable shape
            scaled_attn_logits = scaled_attn_logits.masked_fill(
                ~key_mask,
                torch.finfo(scaled_attn_logits.dtype).min
            )

        if causal_mask:
            mask = torch.triu(
                torch.ones([seq_len, seq_len],
                           dtype=torch.bool,
                           device=x.device),
                diagonal=1
            )  # get mask, [L, L]
            scaled_attn_logits = scaled_attn_logits.masked_fill(
                mask[None, None, :, :], # use mask to get positions where to fill with 0
                torch.finfo(scaled_attn_logits.dtype).min
            )

        attn_weights = F.softmax(scaled_attn_logits, dim=-1)
        # attn_weights.shape = [B, H, L, L]

        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, splited_v)
        # attn_output.shape = [B, H, L, Dh]

        concat_output = self.concat_prj(attn_output)
        # concat_output.shape = [B, L, D]

        output = self.proj_dropout(concat_output)
        # output.shape = [B, L, D]

        return output, attn_weights  # for visualization


def test_output():
    torch.manual_seed(0)

    x = torch.randn(2, 3, 512)
    b, seq_len, d_model = x.shape

    mha = MultiHeadSelfAttn(n_heads=8, d_model=d_model)

    padding_mask = torch.tensor([[True, False, False]])
    output, attn_weights = mha(x, padding_mask=padding_mask, causal_mask=True)

    assert output.shape == (2, seq_len, d_model)
    assert attn_weights.shape == (2, 8, seq_len, seq_len)


def test_attn_sum():
    torch.manual_seed(0)

    x = torch.randn(2, 3, 512)
    b, seq_len, d_model = x.shape

    mha = MultiHeadSelfAttn(n_heads=8, d_model=d_model, attn_dropout=0.0, proj_dropout=0.0)
    output, attn_weight = mha(x)

    p_sum = attn_weight.sum(dim=-1)
    expected = torch.ones_like(p_sum)  # not attn_weight

    assert torch.allclose(p_sum, expected, atol=1e-6)


def test_padding_mask():
    torch.manual_seed(0)

    x = torch.randn(2, 4, 512)
    b, seq_len, d_model = x.shape

    mha = MultiHeadSelfAttn(n_heads=8, d_model=d_model, attn_dropout=0.0, proj_dropout=0.0)

    padding_mask = torch.tensor([[True, True, False, False]])
    _, attn_weights = mha(x, padding_mask=padding_mask)

    masked_weights = attn_weights[..., 2:]  # slice ops
    assert torch.allclose(masked_weights, torch.zeros_like(masked_weights), atol=1e-6)


def test_causal_mask():
    torch.manual_seed(0)

    x = torch.randn(2, 4, 512)
    b, seq_len, d_model = x.shape

    mha = MultiHeadSelfAttn(n_heads=8, d_model=d_model, attn_dropout=0.0, proj_dropout=0.0)

    causal_mask = True
    _, attn_weights = mha(x, causal_mask=causal_mask)  # [2, 8, 4, 4]

    future_mask = torch.triu(
        torch.ones([seq_len, seq_len], dtype=torch.bool),
        diagonal=1
    )  # [4, 4]
    future_weights = attn_weights[:, :, future_mask]  # slice ops, [2, 8, 6]
    future_weights2 = attn_weights[..., future_mask]

    assert torch.allclose(
        future_weights,
        torch.zeros_like(future_weights),
        atol=1e-6
    )


def test_backward():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 512, requires_grad=True)
    b, seq_len, d_model = x.shape

    mha = MultiHeadSelfAttn(n_heads=8, d_model=d_model)
    output, _ = mha(x)

    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def main():
    # test_output()
    # test_attn_sum()
    # test_padding_mask()
    # test_causal_mask()
    test_backward()
    pass


if __name__ == '__main__':
    main()
