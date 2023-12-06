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


def NN(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    layers =  nn.Sequential(
        nn.Linear(dim, hidden_dim, device=nd.cpu(), quantization=True), 
        nn.ReLU(), 
        nn.Linear(hidden_dim, hidden_dim//2,  device=nd.cpu(), quantization=True), 
        nn.ReLU(),
        nn.Linear(hidden_dim//2, num_classes, device=nd.cpu(), quantization=True)
    )

    return layers
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):

    np.random.seed(4)

    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    
    # Calcuate error rate
    total_correct = 0
    total_values = 0
    

    # Calculate avg loss
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
        #x = ndl.Tensor(nd.NDArray(x.numpy(), device = ndl.cpu('int8')), device = ndl.cpu('int8'), dtype='int8')
        #y = ndl.Tensor(nd.NDArray(y.numpy(), device = ndl.cpu('int8')), device = ndl.cpu('int8'), dtype='int8')

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

    # print(total_correct/total_values)
    return total_correct/total_values, loss_values/total_idx, model
    ### END YOUR SOLUTION

def get_quantized_model(model):
    for module in model.modules:
      if isinstance(module, nn.Linear):
        module.quantize()
    return model

def test_quantization(dataloader, model):

    quantized_model = get_quantized_model(model) # change this model to the quantized version 

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

    print(total_correct_quantized/total_values)

    return total_correct_quantized/total_values


def train_mnist(
    batch_size=100, #data
    epochs=10, #training_loop
    optimizer=ndl.optim.Adam, #optimizer
    lr=0.001, #optimizer
    weight_decay=0.001, #optimizer
    hidden_dim=100, #MLP
    data_dir="../data", #data
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION

    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz"
    )

    train_dataloader = ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    
    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    model = NN(784, hidden_dim)


    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)


    training_error_rates = []
    test_error_rates = []


    for i in range(0, epochs):
        training_error_rate, training_avg_loss, model = epoch(train_dataloader, model, opt)
        test_error_rate, test_avg_loss, model = epoch(test_dataloader, model, None)


        training_error_rates.append(training_error_rate)
        test_error_rates.append(test_error_rate)

        print(f"training_error_rate: {training_error_rate} \n training_avg_loss {training_avg_loss} \n test_error_rate {test_error_rate} \n test_avg_loss {test_avg_loss}")

    ############### QUANTIZATION 

    test_quantization(test_dataloader, model)

    ############### QUANTIZATION 

    print(training_error_rates)
    print(test_error_rates)


    return training_error_rate, training_avg_loss, test_error_rate, test_avg_loss

    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(250, 2, ndl.optim.SGD, 0.001, 0.01, 100, "./data")
