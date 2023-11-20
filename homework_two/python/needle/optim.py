"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        if self.momentum != 0:
            for p in range(0, len(self.params)):
                if self.weight_decay != 0:
                    grad = (self.params[p].grad.detach().cached_data + self.weight_decay * self.params[p].detach().cached_data)
                else:
                    grad = self.params[p].grad.detach().cached_data
                
                if p not in self.u.keys():
                    self.u[p] = (1 - self.momentum) * grad
                else:
                    self.u[p] = (self.momentum) * self.u[p] + (1 - self.momentum) * grad

        for p in range(0, len(self.params)):
            if self.momentum != 0:
                grad = self.u[p]
            else:
                if self.weight_decay != 0:
                    grad = self.params[p].grad.detach().cached_data + self.weight_decay * self.params[p].detach().cached_data
                else:
                    grad = self.params[p].grad.detach().cached_data

            self.params[p].cached_data = self.params[p].cached_data - self.lr * grad


    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1

        for p in range(0, len(self.params)):

            if self.weight_decay != 0:
                grad = (self.params[p].grad.detach().cached_data + self.weight_decay * self.params[p].detach().cached_data)
            else:
                grad = self.params[p].grad.detach().cached_data

            if self.t == 1:
                self.m[p] = (1 - self.beta1) * grad
            else:
                self.m[p] = (self.beta1 * self.m[p]) + (1 - self.beta1) * grad

            if self.t == 1:
                self.v[p] = (1 - self.beta2) * (grad ** 2)
            else:
                self.v[p] = (self.beta2 * self.v[p]) + (1 - self.beta2) * (grad ** 2)
            
        
        for p in range(0, len(self.params)):
            mhat = self.m[p] / (1 - self.beta1 ** self.t)
            vhat = self.v[p] / (1 - self.beta2 ** self.t)

            self.params[p].cached_data = self.params[p].cached_data - self.lr * (mhat / ((vhat ** 0.5) + self.eps)) 

        ### END YOUR SOLUTION
