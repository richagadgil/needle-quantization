import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os
from needle import backend_ndarray as nd

import sys

sys.path.append("./python")
sys.path.append("./apps")

np.random.seed(0)
MY_DEVICE = ndl.backend_selection.cuda()

from pympler import asizeof
import argparse

import matplotlib.pyplot as plt


def NN(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    layers =  nn.Sequential(
        nn.Linear(dim, hidden_dim, device=nd.cpu(), quantization=True), 
        nn.ReLU(), 
        nn.Linear(hidden_dim, hidden_dim//2,  device=nd.cpu(), quantization=True), 
        nn.ReLU(),
        nn.Linear(hidden_dim//2, num_classes, device=nd.cpu(), quantization=True)
    )

    return layers


def epoch(dataloader, model, opt=None):

    loss_func = nn.SoftmaxLoss()
    total_correct = 0
    total_values = 0
    loss_values = 0
    total_idx = 0

    if opt:
        model.train()
    else:
        model.eval()

    for idx, batch in enumerate(dataloader):

        if opt:
            opt.reset_grad()

        x = batch[0]
        y = batch[1]
        x = ndl.Tensor(nd.NDArray(x.numpy(), device = ndl.cpu()), device = ndl.cpu())
        y = ndl.Tensor(nd.NDArray(y.numpy(), device = ndl.cpu()), device = ndl.cpu())

        out = model(x)
        loss = loss_func(out, y)

        #### CALCULATE LOSS 
        loss_values += loss.numpy()
        total_idx += 1
        ####

        if opt:
          loss.backward()
          opt.step()

        ##### CALCULATE ERROR RATE 
        check_values = np.argmax(out.numpy(), axis=1) != y.numpy()
        total_correct += np.sum(check_values) 
        total_values += len(check_values)
        #####

    return total_correct/total_values, loss_values/total_idx, model

def get_quantized_model(model):
    for module in model.modules:
      if isinstance(module, nn.Linear):
        module.quantize()
    return model

def test_quantization(dataloader, quantized_model):

    quantized_model.eval()
    total_values = 0
    total_correct_quantized = 0

    for idx, batch in enumerate(dataloader):
        x = batch[0]
        y = batch[1]
        x = ndl.Tensor(nd.NDArray(x.numpy(), device = ndl.cpu()), device = ndl.cpu())
        y = ndl.Tensor(nd.NDArray(y.numpy(), device = ndl.cpu()), device = ndl.cpu())

        out_quantized = quantized_model(x)
        
        check_values = np.argmax(out_quantized.numpy(), axis=1) != y.numpy()
        total_correct_quantized += np.sum(check_values) 
        total_values += len(check_values)

        print(f"Quantized model error rate [Batch {idx+1}]: {total_correct_quantized/total_values}")

        if idx > 10:
          break

    print("\n\n")

    print(f"Final quantized model error rate: {total_correct_quantized/total_values}")
    return total_correct_quantized/total_values


def train_mnist(
    batch_size=100, #data
    epochs=10, #training_loop
    optimizer=ndl.optim.Adam, #optimizer
    lr=0.001, #optimizer
    weight_decay=0.001, #optimizer
    hidden_dim=100, #MLP
    data_dir="../data", #data
    memory_usage=False
):

    train_dataset = ndl.data.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = ndl.data.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    model = NN(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    training_error_rates = []
    test_error_rates = []

    
    if memory_usage:
      print(f"Size of unquantized model pre-training : {asizeof.asizeof(model)} bytes \n\n")

    for i in range(0, epochs):
        training_error_rate, training_avg_loss, model = epoch(train_dataloader, model, opt)
        test_error_rate, test_avg_loss, model = epoch(test_dataloader, model, None)
        training_error_rates.append(training_error_rate)
        test_error_rates.append(test_error_rate)
        print(f"Training Error Rate [Epoch {i+1}]: {training_error_rate}") 
        print(f"Test Error Rate [Epoch {i+1}]: {training_error_rate}") 

    ############### QUANTIZATION 

    if memory_usage:
      print("\n\n")
      print(f"Size of unquantized model post-training : {asizeof.asizeof(model)} bytes \n\n")

    quantized_model = get_quantized_model(model) # change this model to the quantized version 
    
    if memory_usage:
      print(f"Size of quantized model: {asizeof.asizeof(quantized_model)} bytes \n\n")
    else:
      quantized_error = test_quantization(test_dataloader, quantized_model)

      # Length of the arrays
      length = len(training_error_rates)

      # Plotting the training and test error rates
      plt.plot(range(length), training_error_rates, label='Training Error', marker='o')
      plt.plot(range(length), test_error_rates, label='Test Error', marker='o')

      # Plotting the quantized error as a point
      plt.plot(length - 1, quantized_error, 'ro', label='Quantized Error')

      # Adding titles and labels
      plt.title('Training vs Test Error Rates with Quantized Error')
      plt.xlabel('Epochs')
      plt.ylabel('Error Rate')
      plt.xticks(range(length))
      plt.legend()

      # Save the plot
      plt.savefig('plot.png')

      return training_error_rates, test_error_rates, quantized_error

  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a MNIST model.')

    # Adding the arguments
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for the optimizer')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension size for MLP')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for the dataset')
    parser.add_argument('--memory_usage', action='store_true', help='Flag to track memory usage')
    

    args = parser.parse_args()

    train_mnist(
        batch_size=args.batch_size,
        epochs=args.epochs,
        optimizer=ndl.optim.SGD,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        data_dir=args.data_dir,
        memory_usage=args.memory_usage
    )




