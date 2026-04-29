import torch

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_u):
        ctx.lambda_u = lambda_u
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and scale by lambda_u
        lambda_u = ctx.lambda_u
        grad_input = grad_output.clone()
        # Autograd requires returning gradients for all inputs of forward()
        # So we return grad for x, and None for lambda_u
        return -lambda_u * grad_input, None

class GRL(torch.nn.Module):
    def __init__(self, lambda_u=1.0):
        super(GRL, self).__init__()
        self.lambda_u = lambda_u

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_u)
