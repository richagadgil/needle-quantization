import numpy as np
from needle import backend_ndarray as nd


def test_quant():
    shape = (3, 4)
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)

    device = nd.cpu()
    A = nd.array(_A, device=device, dtype="float32")
    B = nd.array(_B, device=device, dtype="float32")
    summation = A + B
    print(summation)

    device = nd.cpu(dtype="int8")
    A_quant = nd.array(_A, device=device, dtype="int8")
    B_quant = nd.array(_B, device=device, dtype="int8")
    summation_quant = A_quant + B_quant
    print(summation_quant)


if __name__ == "__main__":
    test_quant()
