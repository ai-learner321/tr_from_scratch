import torch
import torch.nn as nn


class MyLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(MyLayerNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        output = self.weight * norm_x + self.bias
        return output


def tst_layer_norm_forward():
    torch.manual_seed(0)

    x = torch.randn([2, 5, 512])
    b, seq_len, d_model = x.shape

    eps = 1e-5

    my_layer_norm = MyLayerNorm(d_model, eps=eps)
    my_output = my_layer_norm(x)

    torch_layer_norm = nn.LayerNorm(d_model, eps=eps)
    with torch.no_grad():
        torch_layer_norm.weight.copy_(my_layer_norm.weight)
        torch_layer_norm.bias.copy_(my_layer_norm.bias)

    torch_output = torch_layer_norm(x)

    assert torch.allclose(my_output, torch_output, rtol=1e-5, atol=1e-6)


def tst_layer_norm_backward():
    torch.manual_seed(0)

    x = torch.randn([2, 5, 512], requires_grad=True)
    b, seq_len, d_model = x.shape
    eps = 1e-5
    my_layer_norm = MyLayerNorm(d_model, eps=eps)
    my_output = my_layer_norm(x)

    loss = my_output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert my_layer_norm.weight.grad is not None
    assert torch.isfinite(my_layer_norm.weight.grad).all()

    assert my_layer_norm.bias.grad is not None
    assert torch.isfinite(my_layer_norm.bias.grad).all()


def main():
    # tst_layer_norm_forward()
    tst_layer_norm_backward()
    pass


if __name__ == '__main__':
    main()
