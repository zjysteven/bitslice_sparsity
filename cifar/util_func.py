import torch

class Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.floor()

    @staticmethod
    def backward(ctx, g):
        return g


# https://discuss.pytorch.org/t/how-to-override-the-gradients-for-parameters/3417/6
class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g
