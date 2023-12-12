import sys
sys.path.append('/Users/hwijeen/Documents/CMU/10714_dlsys/needle-quantization/hw4/python')
sys.path.append('/Users/hwijeen/Documents/CMU/10714_dlsys/needle-quantization/hw4/apps')

import numpy as np
from needle import backend_ndarray as nd

def quantize(x, s, z):
    x_q = x + x
    return nd.array(x_q, device=nd.cpu(dtype="int8"), dtype="int8")

if __name__ == "__main__":

    _x = np.random.randn(2,3)
    x = nd.array(_x, dtype="float32")
    print(f"x: {x}")
    x_q = quantize(x, None, None)
    print(f"x_q: {x_q}")
    x_q_numpy = np.array(_x+_x, dtype="int8")
    print(f"x_q_numpy: {x_q_numpy}")
