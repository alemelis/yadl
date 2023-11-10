from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from yadl.activation import ActivationFunction


class Layer:
    weights: np.ndarray
    bias = np.ndarray

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, error, lr):
        raise NotImplementedError

    def randomize(self):
        pass


@dataclass
class Linear(Layer):
    input_size: int
    output_size: int

    def __post_init__(self):
        self.randomize()

    def randomize(self):
        self.weights = np.random.rand(self.input_size, self.output_size) - 0.5
        self.bias = np.random.rand(1, self.output_size) - 0.5

    def forward(self, x):
        self.input = x
        self.output = x.dot(self.weights) + self.bias
        return self.output

    def backward(self, error, lr):
        input_error = error.dot(self.weights.T)
        weights_error = self.input.T.dot(error)

        self.weights -= lr * weights_error
        self.bias -= lr * error
        return input_error

    def __str__(self):
        return f"Linear: {self.input_size}x{self.output_size}"


@dataclass
class Activation(Layer):
    activation_func: ActivationFunction

    def __post_init__(self):
        self.act = self.activation_func()

    def forward(self, x):
        self.input = x
        self.output = self.act.forward(x)
        return self.output

    def backward(self, error, lr=None):
        return self.act.grad(self.input) * error

    def __str__(self):
        return f" ┗━ {str(self.act)}"
