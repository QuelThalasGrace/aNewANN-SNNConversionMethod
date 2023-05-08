import torch
import torch.nn as nn
from torch.autograd import gradcheck, Variable


class steprelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta, steps_num):
        if x.requires_grad:
            ctx.save_for_backward(x)
        ctx.theta = theta
        ctx.steps_num = steps_num

        output = x.clone()
        output[output <= 0] = 0
        for i in range(steps_num):
            index = (theta*i < output) & (output <= theta*(i+1))
            output[index] = i * theta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()

        if ctx.needs_input_grad[0]:
            grad_input[ctx.saved_tensors[0] <= 0] = 0

        return grad_input, None

class StepRelu(nn.Module):
    def __init__(self, theta, steps_num):
        super(StepRelu, self).__init__()

        self.theta = theta
        self.steps_num = steps_num

    def forward(self, x):
        return steprelu.apply(x, self.theta, self.steps_num)

