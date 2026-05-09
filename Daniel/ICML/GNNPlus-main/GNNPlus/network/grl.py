import torch

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_u):
        ctx.lambda_u = lambda_u
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # We multiply by -lambda_u to reverse the gradient
        return -ctx.lambda_u * grad_output, None

class GRL(torch.nn.Module):
    def __init__(self, lambda_u=0.1):
        super().__init__()
        self.lambda_u = lambda_u

    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_u)
