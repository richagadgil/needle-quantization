from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION

        # Compute the maximum value of Z across the specified axes
        m_keep_dims = array_api.max(Z, axis=self.axes, keepdims=True)
        m = array_api.max(Z, axis=self.axes)

        # Subtract the maximum value from Z to avoid overflow in exp(Z - m)
        z_minus_m = array_api.sum(array_api.exp(Z - m_keep_dims), axis=self.axes)
        
        # Compute the sum of exponentials and then the logarithm
        logsumexp = array_api.log(z_minus_m) + m

        return logsumexp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
  
        Z = node.inputs[0]
        i = node.inputs[0].shape

        exp_sum = exp(logsumexp(Z, axes=self.axes))

        if self.axes and not isinstance(self.axes, tuple):
          axes = (self.axes,)
        elif self.axes:
          axes = self.axes
        else:
          axes = range(0, len(i))

        new_shape = list(i)
        for ax in axes:
            new_shape[ax] = 1
        new_shape = tuple(new_shape)

        return reshape(out_grad, new_shape) * (exp(Z) / reshape(exp_sum, new_shape))

      

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

