#from torch import quantization
import torch
import numpy as np
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver
# C, L = 3, 4
# normal = torch.distributions.normal.Normal(0,1)
# inputs = [normal.sample((C, L)), normal.sample((C, L))]
# print(inputs)
weight = np.array([[-0.3038, 0.0876, 0.2934, -0.1025, 0.1970, 0.0540, -0.0518, 0.1946, 0.2116, 0.1890]])
float_tensor = torch.tensor(weight, dtype=torch.float32)
inputs = [float_tensor]
# >>>>>
# [tensor([[-0.0590,  1.1674,  0.7119, -1.1270],
#          [-1.3974,  0.5077, -0.5601,  0.0683],
#          [-0.0929,  0.9473,  0.7159, -0.4574]]]),

# tensor([[-0.0236, -0.7599,  1.0290,  0.8914],
#          [-1.1727, -1.2556, -0.2271,  0.9568],
#          [-0.2500,  1.4579,  1.4707,  0.4043]])]

observers = [MinMaxObserver(), MovingAverageMinMaxObserver(), HistogramObserver()]
for obs in observers:
  for x in inputs: obs(x) 
  print(obs.__class__.__name__, obs.calculate_qparams())

torch.backends.quantized.engine = 'qnnpack'

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1, bias=False)


    def forward(self, x):
        # during the convert step, this will be replaced with a
        # `quantize_per_tensor` call
        x = self.layer(x)
        return x

m = model()

weight = np.array([[-0.3038, 0.0876, 0.2934, -0.1025, 0.1970, 0.0540, -0.0518, 0.1946, 0.2116, 0.1890]])
with torch.no_grad():
    m.layer.weight = torch.nn.Parameter(torch.from_numpy(weight).float())

model_dynamic_quantized = torch.quantization.quantize_dynamic(
    m, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)

q_tensor = model_dynamic_quantized.layer.weight().int_repr().numpy()


B_q = 127
a_q = -128

def affine_quantization(B, a, value):
    S = (B - a) / (B_q - a_q)  # (B - a) / (B_q - A_q) 
    zero_point = int(-np.round((a * B_q - B * a_q) / (B - a))) # (aB_q - Ba_q) / (B - a)
    quantized_value = np.round(value / S + zero_point)  # round(r / S + Z)
    quantized_value[quantized_value > B_q] = B_q
    quantized_value[quantized_value < a_q] = a_q
    quantized_value = quantized_value.astype(np.int8)

    # float_tensor = torch.tensor(value, dtype=torch.float32)
    # q_tensor = torch.quantize_per_tensor(float_tensor, S, zero_point, dtype=torch.qint8).int_repr().numpy()
    # print(np.array_equal(quantized_value, q_tensor))

    return quantized_value

W_B = np.max(weight)
W_A = np.min(weight)
quantized_value = affine_quantization(W_B, W_A, weight)

print(quantized_value - q_tensor) # SLIGHTLY DIFFERENT -- [[0 2 4 1 3 2 2 3 3 4]]

