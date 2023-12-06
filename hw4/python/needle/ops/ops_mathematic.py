"""Operator implementations."""

from abc import ABCMeta
from numbers import Number
from types import new_class
from typing import Concatenate, Optional, List, Tuple, Union

from numpy.core import shape_base

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        i = node.inputs[0]
        return (self.scalar * (i ** (self.scalar - 1) * out_grad), )
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # import pdb; pdb.set_trace()
        lhs, rhs = node.inputs
        return out_grad / rhs, -(out_grad * lhs) / (rhs**2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar, )
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes


    # reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, axes - tuple)
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = list(range(0, len(a.shape)))
        if self.axes:
          pos1 = self.axes[0]
          pos2 = self.axes[1]
        else:
          pos1 = axes[-1]
          pos2 = axes[-2]

        axes[pos1], axes[pos2] = axes[pos2], axes[pos1]
        axes = tuple(axes)
        return a.compact().permute(new_axes=axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        i = node.inputs[0].shape
        return reshape(out_grad, i)
        ### END YOUR SOLUTION



def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad : Tensor, node : Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # i = node.inputs[0].shape
        # new_axis = []
        # n = 0
        # for axis in range(len(self.shape)):
        #     if n >= len(i):
        #         new_axis.append(axis)
        #         continue
        #     if self.shape[axis] != i[n]:
        #         new_axis.append(axis)
        #     if self.shape[axis] == i[n] or i[n] == 1:
        #         n += 1

        # new_axis = tuple(new_axis)
        
        # summed_grad = summation(out_grad, new_axis)
        # if summed_grad.shape != i:
        #   return (reshape(summed_grad, shape = i),) 
        # else:
        #   return (summed_grad,)

        input_shape = node.inputs[0].shape
        output_shape = out_grad.shape
        broadcasted_axes = []
        if len (input_shape) != len(output_shape):
          num_new_axes = len(output_shape) - len(input_shape)
          broadcasted_axes = list(range(num_new_axes))
        added_axes = len(broadcasted_axes)

        for i in range(len(input_shape)):
          if input_shape[-i-1] != output_shape[-i-1]:
            broadcasted_axes.append(len(output_shape) - i - 1)

        # broadcasted axes += tuole(added axes + i for i in range(len(inout shape)) if inout shapelil l= outout shapelil)
        # print(input_shape, output_shape, broadcasted_axes, out_grad.shape)

        answer = summation(out_grad, axes=tuple(broadcasted_axes)).reshape(input_shape)

        return answer

        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return array_api.sum(a, axis=self.axes)

        if self.axes is None:
          return a.sum()
        if isinstance(self.axes, int):
          self.axes = (self.axes,)
        
        sorted_axes = sorted(self.axes, reverse=True)

        for ax in sorted_axes:
          a = a.sum(axis=ax)
      
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        i = node.inputs[0].shape

        if self.axes is not None and not isinstance(self.axes, tuple):
          axes = (self.axes,)
        elif self.axes is not None:
          axes = self.axes
        else:
          axes = range(0, len(i))

        new_shape = list(i)
        for ax in axes:
            new_shape[ax] = 1
        new_shape = tuple(new_shape)

      
        grad = reshape(out_grad, new_shape)
        answer = (broadcast_to(grad, i), )
        return answer
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = out_grad @ transpose(rhs)
        rhs_grad = transpose(lhs) @ out_grad

        while lhs_grad.shape != lhs.shape:
          lhs_grad = summation(lhs_grad, axes=0)
        while rhs_grad.shape != rhs.shape:
          rhs_grad = summation(rhs_grad, axes=0)
        
        return lhs_grad, rhs_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (negate(out_grad), ) 
        ### END YOUR SOLUTION

def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * power_scalar(node.inputs[0], -1), )
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * exp(node.inputs[0]), )
        ### END YOUR SOLUTION



def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        i = node.inputs[0].realize_cached_data() 
        #new_grad = mul_scalar(out_grad, 1.0)
        #print(i)
        #print(i * (i > 0))
        #out_grad.cached_data[i < 0] = 0
        return out_grad * (i > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        tanh_output = node
        tanh_grad = 1 + negate(tanh_output ** 2)
        answer = (out_grad * tanh_grad, )

        return answer
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis
        

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = args[0].shape

        # if len(shape) == 1:
        #   import pdb; pdb.set_trace()
        #   args = [array_api.reshape(a, (1, a.shape[0])) for a in args]

        result_shape = list(shape)
        result_shape.insert(self.axis, len(args))

        result = array_api.full(result_shape, 0, device=args[0].device)

        # Step 3: Copy elements
        for i, tensor in enumerate(args):
            # Using numpy's advanced slicing to insert the arrays
            indices = [slice(None)] * len(result_shape)
            indices[self.axis] = i
            result[tuple(indices)] = tensor.compact()

        return result
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # l = []
        # for tensor in node.inputs[0]:
        #   l.append(power_scalar(tensor, 0))
        # answer = make_tuple(*l)
        # return answer
        # import pdb; pdb.set_trace() #[array([[ 1.23020151,  1.20236437, -0.38729874
        answer = split(out_grad, axis=self.axis)
        return answer
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))

class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION

        A = A.compact()

        split_shape = []
        for i in range(0, len(A.shape)):
          if i != self.axis:
            split_shape.append(A.shape[i])

        split_shape = tuple(split_shape)

        splits = []
        for i in range(A.shape[self.axis]):
            # Using numpy indexing to split the tensor
            indices = [slice(None)] * len(A.shape)
            indices[self.axis] = i

            split_section = A[tuple(indices)].compact().reshape(split_shape)
            splits.append(split_section)

        return tuple(splits)
 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #import pdb; pdb.set_trace()
        #return out_grad
        answer = stack(out_grad, axis=self.axis)
        return answer
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, axes=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (flip(out_grad, axes=self.axes),)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)

        for i in self.axes:
          if (i >= len(shape)):
            return a
          shape[i] = (self.dilation + 1) * shape[i]

        new_array = array_api.full(shape, 0, device=a.device) #numpy.zeros(shape)
        slices = tuple(slice(0, new_array.shape[i], self.dilation+1) if i in self.axes else slice(None) for i in range(len(new_array.shape)))
        new_array[slices] = a
        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        slices = tuple(slice(0, out_grad.shape[axis], self.dilation + 1) if axis in self.axes else slice(None) for axis in range(len(out_grad.shape)))
        out_grad.cached_data = out_grad.realize_cached_data()[slices]
        return out_grad
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION

        # batch_size, height, width, channels_in = A.shape
        # kernel_size, _, channels_in, channels_out = B.shape

        # A_padded = A.pad(((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))

        # output_height = (height - kernel_size + 2 * self.padding) // self.stride + 1
        # output_width = (width - kernel_size + 2 * self.padding) // self.stride + 1

        # output = array_api.full((batch_size, output_height, output_width, channels_out), 0, device=A.device) 

        # B = B.permute((3, 0, 1, 2))

        # for n in range(batch_size):
        #     for h in range(output_height):
        #         for w in range(output_width):
        #             for c in range(channels_out):
        #                 h_start = h * self.stride
        #                 h_end = h_start + kernel_size
        #                 w_start = w * self.stride
        #                 w_end = w_start + kernel_size

        #                 output[n, h, w, c] = array_api.sum(
        #                     A_padded[n, h_start:h_end, w_start:w_end, :] * B[c, :, :, :]
        #                 )

        # N,H,W,C_in = Z.shape
        # K,_,_,C_out = weight.shape
        # out = np.zeros((N,H-K+1,W-K+1,C_out))
        
        # for i in range(kernel_size):
        #     for j in range(kernel_size):
        #         #import pdb; pdb.set_trace()
        #         output += A[:, i:i+output_height ,j:j+output_width, :] @ B[i, j, :, :].permute((2,3,0,1))

        # return output 

        ### BEGIN YOUR SOLUTION
        #A.shape = N x H x W X C_in
        #B.shape = K x K x C_in x C_out
        #out.shape = N x H_out x W_out x C_out
        ### END YOUR SOLUTION
        batch_size, height, width, channels_in = A.shape
        kernel_size, _, _, channels_out = B.shape
        output_height = (height + 2 * self.padding - kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - kernel_size) // self.stride + 1

        A_padded = A.pad( ((0,0),(self.padding,self.padding),(self.padding,self.padding), (0,0)) )

        A_strided = A_padded.permute( (0, 3, 1, 2) ).compact() # batch_size x channels_in x height x width
        # A_strided = A_strided.pad( ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)) )

        _, _, height_strided, width_strided = A_strided.shape
        A_strided = A_strided.as_strided( shape = (batch_size, channels_in, output_height, output_width, kernel_size, kernel_size), 
                                strides = ( height_strided*width_strided*channels_in, height_strided*width_strided, width_strided*self.stride, self.stride, width_strided, 1) )
        A_strided = A_strided.permute( (0, 2, 3, 4, 5, 1) ).compact().reshape( (batch_size * output_height * output_width, kernel_size * kernel_size * channels_in) ).compact()
        B = B.compact().reshape( (kernel_size * kernel_size * channels_in, channels_out) ).compact()
        r = A_strided @ B
        return r.reshape( (batch_size, output_height, output_width, channels_out) )




    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        
        # X, W = node.inputs
        # # X: (3, 14, 14, 8) = batch_size, height, width, channels_in
        # # W: (3, 3, 8, 16) = kernel_size, _, channels_in, channels_out

        # # X_grad: (3, 14, 14, 8)
        # # W_grad: (3, 3, 8, 16)

        # # out_grad: (3, 12, 12, 16)

        # # X_grad = ≈conv(≈out_grad, ≈W) | conv((3, 12, 12, 16), (3, 3, 8, 16))

        
        # W_rot180 = transpose(flip(W, axes=(0, 1)), axes=(2, 3))  # Rotate the weights by 180 degrees

        # if self.stride > 1:
        #   out_grad = dilate(out_grad, axes=(1,2), dilation = self.stride - 1)

        # padding = X.shape[1] - out_grad.shape[1]

        # if X.shape[1] - out_grad.shape[1] > 0:
        #   padding = X.shape[1] - out_grad.shape[1]
        # elif X.shape[1] - out_grad.shape[1] == 0:
        #   padding = 1
        # elif X.shape[1] - out_grad.shape[1] < 0:
        #   padding = 0

        # X_grad = conv(out_grad, W_rot180, padding=padding)

        
        

        ### BEGIN YOUR SOLUTION
        batch_size, output_height, output_width, channels_out = out_grad.shape
        X,W = node.inputs
        batch_size, height, width, channels_in = X.shape
        kernel_size, _, _, channels_out = W.shape

        # X.grad 
        W_transposed = transpose(flip(W, axes=(0, 1)), axes=(2, 3))
        
        outgrad_dilated = dilate(out_grad, axes=(1,2), dilation = self.stride - 1)
        X_grad = conv(outgrad_dilated, W_transposed, padding = kernel_size - self.padding - 1) #, stride=1)

        # W.grad = conv(X, out_grad)
        outgrad_tranposed = transpose(transpose(outgrad_dilated, axes=(0,1)), (1,2))
        X_transposed = transpose(X, (0,3))

       # import pdb; pdb.set_trace()

        W_grad = transpose(transpose(conv(X_transposed, outgrad_tranposed, padding = self.padding), (0,1)), (1,2))


        return X_grad, W_grad


        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


class Clip(TensorOp):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def compute(self, a):
        # Clip the values in tensor 'a' to be within the range [min_val, max_val]
        return array_api.clip(a, self.min_val, self.max_val)

    def gradient(self, out_grad, node):
        # The gradient of the clip operation is 1 for values within the range
        # and 0 for values outside the range. It acts like a mask.
        a = node.inputs[0].realize_cached_data()
        grad_mask = (a >= self.min_val) & (a <= self.max_val)
        return out_grad * grad_mask


def clip(a, min_val, max_val):
    return Clip(min_val, max_val)(a)

