import torch
import torch.nn as nn

from tr_from_scratch.layer_norm import MyLayerNorm


class AddNorm(nn.Module):
    def __init__(self, d_model=512, eps=1e-5):
        super(AddNorm, self).__init__()
        self.d_model = d_model

        self.norm = MyLayerNorm(d_model, eps=eps)

    def forward(self, x, sublayer_output):
        if x.shape != sublayer_output.shape:
            raise ValueError(f"x shape {x.shape} does not match sublayer_output "
            f"shape {sublayer_output.shape}")

        norm_x = self.norm(x + sublayer_output)
        return norm_x


def test_add_norm_against_torch():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 512)
    b, seq_len, d_model = x.shape

    sublayer_output = torch.randn(2, 3, 512)

    my_addnorm = AddNorm(d_model)
    torch_addnorm = nn.LayerNorm(d_model)

    with torch.no_grad():
        torch_addnorm.weight.copy_(my_addnorm.norm.weight)
        torch_addnorm.bias.copy_(my_addnorm.norm.bias)

    my_output = my_addnorm(x, sublayer_output)
    torch_output = torch_addnorm(x + sublayer_output)

    assert torch.allclose(my_output, torch_output, atol=1e-6, rtol=1e-5)


def test_add_norm_0_sublayer_output():
    torch.manual_seed(0)

    x = torch.randn(2, 3, 512)
    b, seq_len, d_model = x.shape
    sublayer_output = torch.zeros_like(x)

    add_norm = AddNorm(d_model)
    output = add_norm(x, sublayer_output)

    norm_ouput = add_norm.norm(x)
    assert torch.allclose(output, norm_ouput, atol=1e-6, rtol=1e-5)


def test_add_norm_0_residual():
    torch.manual_seed(0)
    sublayer_output = torch.randn(2, 3, 512)
    b, seq_len, d_model = sublayer_output.shape

    x = torch.zeros_like(sublayer_output)

    add_norm = AddNorm(d_model)
    output = add_norm(x, sublayer_output)

    norm_output = add_norm.norm(sublayer_output)

    assert torch.allclose(output, norm_output, atol=1e-6, rtol=1e-5)


def test_add_norm_shape():
    torch.manual_seed(0)

    x = torch.randn(2, 3, 512)
    b, seq_len, d_model = x.shape
    sublayer_output = torch.randn(2, 3, 512)
    add_norm = AddNorm(d_model)

    output = add_norm(x, sublayer_output)
    assert output.shape == x.shape


def test_add_norm_shape_mismatch():
    import pytest

    torch.manual_seed(0)

    x = torch.randn(2, 3, 512)
    b, seq_len, d_model = x.shape
    sublayer_output = torch.randn(2, 1, 512)

    add_norm = AddNorm(d_model)

    with pytest.raises(ValueError):
        add_norm(x, sublayer_output)


def test_add_norm_backward():
    torch.manual_seed(0)

    x = torch.randn(2, 3, 512, requires_grad=True)
    b, seq_len, d_model = x.shape
    sublayer_output = torch.randn(x.shape, requires_grad=True)

    add_norm = AddNorm(d_model)
    output = add_norm(x, sublayer_output)

    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert sublayer_output.grad is not None
    assert add_norm.norm.weight.grad is not None
    assert add_norm.norm.bias.grad is not None

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(sublayer_output.grad).all()
    assert torch.isfinite(add_norm.norm.weight.grad).all()
    assert torch.isfinite(add_norm.norm.bias.grad).all()


def main():
    # test_add_norm_against_torch()
    # test_add_norm_0_sublayer_output()
    # test_add_norm_0_residual()
    # test_add_norm_shape()
    # test_add_norm_shape_mismatch()
    test_add_norm_backward()
    pass


if __name__ == "__main__":
    main()
