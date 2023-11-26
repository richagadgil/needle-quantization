from pathlib import Path
import sys
needle_path = Path(__file__).parent.parent.parent.joinpath("python").as_posix()
sys.path.append(needle_path)

import numpy as np
import pytest
import needle as ndl
from needle import backend_ndarray as nd


_DEVICES = [
    nd.cpu(),
    pytest.param(
        nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU")
    ),
]


def test_quant():
    device = nd.cpu()
    shape = (3, 4)
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)

    A = nd.array(_A, device=device, dtype="float32")
    B = nd.array(_B, device=device, dtype="float32")
    summation = A + B

    A_quant = nd.array(_A, device=device, dtype="int8")
    B_quant = nd.array(_B, device=device, dtype="int8")
    summation_quant = A_quant + B_quant

    assert np.allclose(summation.asnumpy(), summation_quant.asnumpy(), atol=1e-3)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
