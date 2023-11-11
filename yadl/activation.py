from abc import abstractmethod

import numpy as np


class ActivationFunction:
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def grad(self, x):
        raise NotImplementedError


class Identity(ActivationFunction):
    def forward(self, x):
        return x

    def grad(self, x):
        return 1.0

    def __str__(self):
        return "Identity"


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def grad(self, x):
        return self.forward(x) * (1 - self.forward(x))

    def __str__(self):
        return "Sigmoid"


class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)

    def grad(self, x):
        return 1 - self.forward(x) ** 2

    def __str__(self):
        return "Tanh"
