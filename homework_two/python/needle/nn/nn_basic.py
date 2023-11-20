"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32", quantization=True, quantization_profile="signed_eight", running_max=False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features), dtype=dtype)
        self.bias = Parameter(ops.transpose(init.kaiming_uniform(self.out_features, 1)),  dtype=dtype)
        self.avg_W_B = []
        self.avg_W_a = []
        self.avg_B_B = []
        self.avg_B_a = []

        self.W_B = None
        self.W_A = None
        self.B_B = None
        self.B_a = None

        self.quantized_weight = None 
        self.quantized_bias = None

        self.quantization = quantization

        if quantization_profile == "signed_eight":
            self.B_q = 127
            self.a_q = -128
        elif quantization_profile == "unsigned_eight":
            self.B_q = 255
            self.a_q = 0

        self.running_max = running_max

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        if self.running_max:
            self.avg_W_B.append(np.max(self.weight.detach().numpy()))
            self.avg_W_a.append(np.min(self.weight.detach().numpy()))
            self.avg_B_B.append(np.max(self.bias.detach().numpy()))
            self.avg_B_a.append(np.min(self.bias.detach().numpy()))
            self.W_B = np.mean(self.avg_W_B)
            self.W_A = np.mean(self.avg_W_a)
            self.B_B = np.mean(self.avg_B_B)
            self.B_a = np.mean(self.avg_B_a)
        else:
            self.W_B = np.max(self.weight.detach().numpy()) 
            self.W_A = np.min(self.weight.detach().numpy()) 
            self.B_B = np.max(self.bias.detach().numpy()) 
            self.B_a = np.min(self.bias.detach().numpy()) 

        if self.training:

            y = X @ self.weight
    
            if self.bias:
                y = ops.add(y, ops.broadcast_to(self.bias, y.shape))
        
        else:

            def affine_quantization(B, a, value):
                S = (B - a) / (self.B_q - self.a_q)  # (B - a) / (B_q - A_q) 
                zero_point = -np.round((a * self.B_q - B * self.a_q) / (B - a)) # (aB_q - Ba_q) / (B - a)
                quantized_value = np.round(value / S + zero_point)  # round(r / S + Z)
                quantized_value[quantized_value > self.B_q] = self.B_q
                quantized_value[quantized_value < self.a_q] = self.a_q
                return Tensor(quantized_value)
            

            self.quantized_weight = affine_quantization(self.W_B, self.W_A, self.weight.detach().numpy())
            self.quantized_bias =  affine_quantization(self.B_B, self.B_a, self.bias.detach().numpy())

            if self.quantization:
                y = X @ self.quantized_weight

                if self.bias:
                    y = ops.add(y, ops.broadcast_to(self.bias, y.shape))
            else:
                y = X @ self.weight
    
                if self.bias:
                    y = ops.add(y, ops.broadcast_to(self.bias, y.shape))


        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        n = np.prod(X.shape[1:])
        return ops.reshape(X, (X.shape[0], n))
        #raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size = logits.shape[0] 
        y_one_hot = init.one_hot(logits.shape[1], y)
        z_y = ops.summation(ops.multiply(logits, y_one_hot), axes=1)


        loss = ops.logsumexp(logits, axes=1) - z_y

        average_loss = ops.divide_scalar(ops.summation(loss), batch_size)

        return average_loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim), requires_grad=True, dtype=dtype)
        self.bias = Parameter(init.zeros(self.dim), requires_grad=True, dtype=dtype)
        self.running_mean = init.zeros(self.dim, dtype=dtype)
        self.running_var = init.ones(self.dim, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        def broadcast_reshape(y: Tensor):
            return ops.broadcast_to(ops.reshape(y, (1, x.shape[1])), x.shape)

        if self.training:
            mean = ops.summation(x, axes=(0, )) / x.shape[0]
            var = ops.summation((x - broadcast_reshape(mean)) ** 2, axes=(0, ))  / x.shape[0]

            # detach needed here?
            self.running_mean = self.momentum * mean.detach() + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var.detach() + (1 - self.momentum) * self.running_var
       
        else:
            mean = self.running_mean
            var = self.running_var

        bn = (x - broadcast_reshape(mean)) / broadcast_reshape(ops.power_scalar(var + self.eps, 1 / 2)) 
    
        y = broadcast_reshape(self.weight) * bn + broadcast_reshape(self.bias)


        return y
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.w = Parameter(init.ones(self.dim), dtype=dtype)
        self.b = Parameter(init.zeros(self.dim), dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        def broadcast_reshape(y: Tensor):
            return ops.broadcast_to(ops.reshape(y, (x.shape[0], 1)), x.shape)

        mean = broadcast_reshape(ops.summation(x, axes=(1, ))) / x.shape[1]
        var = broadcast_reshape(ops.summation((x - mean) ** 2, axes=(1, ))) / x.shape[1]

        bn = (x - mean) / (ops.power_scalar(var + self.eps, 0.5))

        # y = ops.broadcast_to(self.w, x.shape) * bn + ops.broadcast_to(self.b, x.shape)
        y = ops.broadcast_to(ops.reshape(self.w, (1, x.shape[1])), x.shape) * bn + ops.broadcast_to(ops.reshape(self.b, (1, x.shape[1])), x.shape)

        return y
        ### END YOUR SOLUTION
        


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, dtype='float32')
            return mask * (x / (1-self.p))
        else:
            return x

        ### END YOUR SOLUTION



class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
