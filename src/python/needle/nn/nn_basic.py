"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import needle as ndl
from needle import backend_ndarray as nd

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
        self, in_features, out_features, bias=True, device=None, dtype="float32", quantization=True, 
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.quantize_input = False

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features), dtype=dtype, device=device)
        if bias == True:
          self.bias = Parameter(ops.transpose(init.kaiming_uniform(self.out_features, 1)),  dtype=dtype, device=device)
        else:
          self.bias = None

        self.quantization = quantization

        if self.quantization:

          self.W_B = None
          self.W_A = None

          self.X_A = None
          self.X_B = None

          self.y_A = None
          self.y_B = None

          self.b_A = None
          self.b_B = None

          self.beta_q = 127
          self.alpha_q = -128

          self.W_q = None
          self.b_q = None
          self.s_W, self.z_W = None, None
          self.s_b, self.z_b = None, None

        ### END YOUR SOLUTION


    #### VIRTUAL QUANTIZATION HELPER ###############################################################################################

    def generate_quantization_constants(self, alpha, beta):

        S = (beta - alpha) / (self.beta_q - self.alpha_q)
        Z = int((beta * self.alpha_q - alpha * self.beta_q) / (beta - alpha))

        return S, Z

    def quantization_tensor(self, x, s, z):

        x_q = 1 / s * x + z
        x_q = ops.clip(x_q, self.alpha_q, self.beta_q)

        x_q = nd.array(x_q.numpy(), device=nd.cpu(dtype="int8"), dtype="int8")
        return Tensor(x_q, device=ndl.cpu(), dtype='int8')

    def dequantization(self, x_q, s, z):

        # x_q - z might go outside the quantization range.

        #x_q_n = x_q.numpy().astype(np.int32)
        x_q_32 = Tensor(nd.array(x_q.numpy(), device = ndl.cpu()), device = ndl.cpu())

        #x_n = s * (x_q_n - z).astype(np.float32)
        x = s * (x_q_32 - z)

        return x

    def quantization_matrix_multiplication_int8(self, X_q, W_q, b_q, s_X, z_X, s_W, z_W,
                                            s_b, z_b, s_Y, z_Y):




        p = W_q.shape[0]

        b_q_32 = Tensor(nd.array(b_q.numpy(), device = ndl.cpu()), device = ndl.cpu())
        W_q_32 = Tensor(nd.array(W_q.numpy(), device = ndl.cpu()), device = ndl.cpu())
        X_q_32 = Tensor(nd.array(X_q.numpy(), device = ndl.cpu()), device = ndl.cpu())

        A = z_Y

        # Part B: Scaling and shifting of b_q
        # B_n = (s_b / s_Y) * (b_q.astype(np.int32) - z_b)
        B = (s_b / s_Y) * (b_q_32 - z_b)

        # Part C: Matrix multiplication of X_q and W_q
        # C_n = np.matmul(X_q.astype(np.int32), W_q.astype(np.int32))
        C = ops.matmul(X_q_32, W_q_32)

        # Part D: Adjustments related to X_q
        # D_n = - z_W * np.sum(X_q.astype(np.int32), axis=1, keepdims=True)
        D = ops.broadcast_to(- z_W * ops.reshape(ops.summation(X_q_32, axes=1), (X_q_32.shape[0], 1)), C.shape)

        # Part E: Adjustments related to W_q
        # E_n = - z_X * np.sum(W_q.astype(np.int32), axis=0, keepdims=True)
        E = ops.broadcast_to(- z_X * ops.reshape(ops.summation(W_q_32, axes=0), (1, W_q_32.shape[1])), C.shape)

        # Part F: Constant term involving z_X and z_W
        F = p * z_X * z_W

        # Final calculation
        # Y_q_simulated_n = A + B_n + ((s_X * s_W / s_Y) * (C_n + D_n + E_n + F))
        Y_q_simulated = A + ops.broadcast_to(B, C.shape) + ((s_X * s_W / s_Y) * (C + D + E + F))


        Y_q_simulated = ops.clip(Y_q_simulated, self.alpha_q, self.beta_q)
        Y_q_simulated = nd.array(Y_q_simulated.numpy(), device=nd.cpu(dtype="int8"), dtype="int8")
        return Tensor(Y_q_simulated, device=ndl.cpu(), dtype='int8')



    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        if self.quantize_input and self.quantization:
          
          #### QUANTIZATION ###############################################################################################


          s_X, z_X = self.generate_quantization_constants(self.X_A, self.X_B)
          s_Y, z_Y = self.generate_quantization_constants(self.y_A, self.y_B)
          

          X_q = self.quantization_tensor(X, s_X, z_X)


          y_q = self.quantization_matrix_multiplication_int8(X_q, self.W_q, self.b_q, s_X, z_X, self.s_W, self.z_W,
                                            self.s_b, self.z_b, s_Y, z_Y)


          y = Tensor(self.dequantization(y_q, s_Y, z_Y), device=ndl.cpu(), dtype='float32')

          #### QUANTIZATION ###############################################################################################
          
        else:
          
          y = X @ self.weight

          if self.bias:
            y = ops.add(y, ops.broadcast_to(self.bias, y.shape))

          self.W_B = np.max(self.weight.detach().numpy()) 
          self.W_A = np.min(self.weight.detach().numpy()) 

          self.X_B = np.max(X.detach().numpy()) 
          self.X_A = np.min(X.detach().numpy()) 

          self.y_B = np.max(y.detach().numpy()) 
          self.y_A = np.min(y.detach().numpy()) 

          self.b_B = np.max(self.bias.detach().numpy()) 
          self.b_A = np.min(self.bias.detach().numpy()) 
            
        return y

    def quantize(self):

        self.quantize_input = True

        self.s_W, self.z_W = self.generate_quantization_constants(self.W_A, self.W_B)
        self.s_b, self.z_b = self.generate_quantization_constants(self.b_A, self.b_B)

        self.W_q = self.quantization_tensor(self.weight, self.s_W, self.z_W)
        self.b_q = self.quantization_tensor(self.bias, self.s_b, self.z_b)
        
        del self.weight
        del self.bias





class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
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
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.w = Parameter(init.ones(self.dim), dtype=dtype, device=device)
        self.b = Parameter(init.zeros(self.dim), dtype=dtype, device=device)
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
            mask = init.randb(*x.shape, p=1-self.p, dtype=x.dtype, device=x.device)
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


