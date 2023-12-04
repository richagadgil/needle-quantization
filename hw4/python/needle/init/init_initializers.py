import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6/(fan_in+fan_out))
    t = rand(fan_in, fan_out, low=-a, high=a)
    return t
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    t = randn(fan_in, fan_out, std=std)
    return t
    ### END YOUR SOLUTION



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if nonlinearity == "relu":
      gain = math.sqrt(2)
    
    bound = gain * math.sqrt(3/fan_in)
    if shape is not None:
      t = rand(*shape, low=-bound, high=bound)
    else:
      t = rand(fan_in, fan_out, low=-bound, high=bound)
    return t
    ### END YOUR SOLUTION

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if nonlinearity == "relu":
      gain = math.sqrt(2)
    
    std = gain / math.sqrt(fan_in)
    if shape is not None:
      t = rand(*shape, low=-bound, high=bound)
    else:
      t = randn(fan_in, fan_out, std=std)
    return t
    ### END YOUR SOLUTION