import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff=2048, p_dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        hidden = self.linear(x)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.linear2(hidden)
        return hidden


def test_output_shape():
    torch.manual_seed(0)

    x = torch.randn(2, 3, 512)
    b, seq_len, d_model = x.shape
    d_ff = 2048
    ffn = PositionWiseFFN(d_model, d_ff=d_ff, p_dropout=0.0)

    output = ffn(x)
    assert output.shape == x.shape


def test_output_shape_2d():
    torch.manual_seed(0)
    x = torch.randn(3, 512)
    seq_len, d_model = x.shape

    d_ff = 2048
    ffn = PositionWiseFFN(d_model, d_ff=d_ff, p_dropout=0.0)
    output = ffn(x)
    assert output.shape == x.shape


def test_forward_matches_manual_computation():
    torch.manual_seed(0)

    x = torch.randn(2, 3, 512)
    b, seq_len, d_model = x.shape
    d_ff = 2048
    ffn = PositionWiseFFN(d_model, d_ff=d_ff, p_dropout=0.0)
    actual = ffn(x)

    hidden = F.linear(x, ffn.linear.weight, ffn.linear.bias)
    hidden = F.relu(hidden)
    hidden = F.dropout(hidden, p=0.0)
    expected = F.linear(hidden, ffn.linear2.weight, ffn.linear2.bias)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_position_independance():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 512)
    x2 = x.clone()

    b, seq_len, d_model = x.shape
    d_ff = 2048
    p_dropout = 0.0
    ffn = PositionWiseFFN(d_model, d_ff=d_ff, p_dropout=p_dropout)

    output = ffn(x)

    x2[0, 2] += 10
    output2 = ffn(x2)

    assert not torch.allclose(output[0, 2], output2[0, 2])

    unchanged_pos = [0, 1]
    assert torch.allclose(output[0, unchanged_pos], output2[0, unchanged_pos],
                          atol=1e-6, rtol=1e-5)

    assert torch.allclose(output[1], output2[1], atol=1e-6, rtol=1e-5)


def test_backward():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 512, requires_grad=True)
    b, seq_len, d_model = x.shape
    d_ff = 2048
    b, seq_len, d_model = x.shape
    d_ff = 2048
    p_dropout = 0.0
    ffn = PositionWiseFFN(d_model, d_ff=d_ff, p_dropout=p_dropout)
    output = ffn(x)

    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, param in ffn.named_parameters():
        assert param.grad is not None
        assert torch.isfinite(param).all()


def test_dropout_change_output_in_train_mode():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 512)
    b, seq_len, d_model = x.shape
    d_ff = 2048
    p_dropout = 0.1

    ffn = PositionWiseFFN(d_model, d_ff=d_ff, p_dropout=p_dropout)
    ffn.train()

    output = ffn(x)
    output2 = ffn(x)

    assert not torch.allclose(output, output2, atol=1e-6, rtol=1e-5)


def test_dropout_disabled_in_eval_mode():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 512)
    b, seq_len, d_model = x.shape
    d_ff = 2048
    p_dropout = 0.1

    ffn = PositionWiseFFN(d_model, d_ff=d_ff, p_dropout=p_dropout)
    ffn.eval()

    output = ffn(x)
    output2 = ffn(x)
    assert torch.allclose(output, output2, atol=1e-6, rtol=1e-5)


def main():
    # test_output_shape()
    # test_output_shape_2d()
    # test_forward_matches_manual_computation()
    # test_position_independance()
    # test_backward()
    # test_dropout_change_output_in_train_mode()
    # test_dropout_disabled_in_eval_mode()
    pass


if __name__ == '__main__':
    main()
