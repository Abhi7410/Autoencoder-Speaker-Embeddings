###################################################
# Domain Adaptation uses Gradient Reversal Layer which is implemented in this file
###############################################

import torch
import torch.nn as nn
from torch.autograd import Function

class GRL(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x) * constant
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None
