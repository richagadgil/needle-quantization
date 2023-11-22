# Supporting quantization on Needle Framework



## Motivation
The escalating size of neural networks demands efficiency improvements in deployment and execution, especially given the impracticality of running state-of-the-art (SOTA) models on academic hardware. Neural network quantization addresses this challenge by significantly reducing the precision of model parameters and activations, diminishing both memory and computational requirements. In fact, many of the bilion-scale models are deployed using various quantization techniques[1]. In this project, we extend the needle framework to support various quentization schemes.

This will be a project that involves both implementation and experimentation. As a final deliverable we will 1) prodive needle quantization API,  2) implement the internals of quantization (fake or real quantization), and performance comparision between different data types in terms of the memory footprint and the accuracy.

## Quantiazation API
```python
from needle import quantization

# case 1. quantizing weights only
quantized_model = model.quantize(data_type="int8") # weight quantization

# case 2. quantizing weights and activation
def calibrate(model, dataloader):
  for batch in dataloader:
    _ = model(batch)  # internally gathers statistics
  return model

quantized_model = quantization.quantize(data_type="int8")
quantized_model = quantization.insert_obserserver(model)
quantized_model = calibrate(model)
```

## Internals: What quantization to support
We focus on supporting post-training quantization. It is a way of converting weights and activations that are in full precision (`float32`) into lower precision data formats like `float16` or `int8`. We plan to have flexibility on where to quantize so that the users can balance the trade between efficiency and accuracy. 
* Weight quantization: We start with quantizing weight that are in `float32` data format to lower precision formats. Specifically, we experiment with `float16`, `int8`, `uint8`.
* Activation quantization: We also allow users to quantize activation as well. We will support two ways on how to calibrate activation: dynamic quantization and static quantization. The difference between the two schemes is on how to select the range of `float32`. Dynamic quantization selectcs the range on the fly at inference time whereas the static methods sets the range before the inference time using the sample data.


# Experiments: What to measure

We will analyze and measure the memory usage of different quantization techniques for both theoretical and empirical purposes in terms of memory footprint and accuracy. It is important to note that the use of fake quantization might be required depending on hardware support, resulting in similar empirical measurements across different data types. To assess accuracy, we compare model outputs obtained with different data types against a reference of 'float32'. The deviation from the reference output is used as a metric.

[1] Dettmers, Tim et al. “LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.” ArXiv abs/2208.07339 (2022): n. pag.



