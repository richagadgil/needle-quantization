import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

import sys

sys.path.append("./python")
sys.path.append("./apps")

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    block1 = nn.Sequential(
        nn.Linear(dim, hidden_dim), 
        norm(hidden_dim), 
        nn.ReLU(), 
        nn.Dropout(drop_prob), 
        nn.Linear(hidden_dim, dim), 
        norm(dim),
    )

    model = nn.Sequential(
        nn.Residual(block1),
        nn.ReLU()
    )

    return model


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    # Initial layer to expand dimension from input dim to hidden_dim
    layers = [nn.Linear(dim, hidden_dim), nn.ReLU()]

    # Add residual blocks
    for _ in range(num_blocks):
        layers.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))

    # # Add final classification layer
    layers.append(nn.Linear(hidden_dim, num_classes))

    return nn.Sequential(*layers)
    ### END YOUR SOLUTION

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
        nn.Linear(dim, hidden_dim, quantization=True), 
        nn.ReLU(), 
        nn.Linear(hidden_dim, hidden_dim//2, quantization=True), 
        nn.ReLU(),
        nn.Linear(hidden_dim//2, num_classes, quantization=True)
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

        out = model(x)
        loss = loss_func(out, y)

        #### CALCULATE LOSS 
        loss_values += loss.cached_data
        total_idx += 1
        ####

        if opt:
            loss.backward()
            opt.step()

        ##### CALCULATE ERROR RATE 
        check_values = np.argmax(out.cached_data, axis=1) != y.cached_data
        total_correct += np.sum(check_values) 
        total_values += len(check_values)
        #####


    return total_correct/total_values, loss_values/total_idx
    ### END YOUR SOLUTION


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


    for i in range(0, epochs):
        training_error_rate, training_avg_loss = epoch(train_dataloader, model, opt)
        test_error_rate, test_avg_loss = epoch(test_dataloader, model, None)

    print(f"training_error_rate: {training_error_rate} \n training_avg_loss {training_avg_loss} \n test_error_rate {test_error_rate} \n test_avg_loss {test_avg_loss}")

    return training_error_rate, training_avg_loss, test_error_rate, test_avg_loss

    ### END YOUR SOLUTION




if __name__ == "__main__":
    train_mnist(250, 2, ndl.optim.SGD, 0.001, 0.01, 100, "../data")
