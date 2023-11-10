import numpy as np


class Loss:
    def forward(self, y, y_hat):
        raise NotImplementedError

    def grad(self, y, y_hat):
        raise NotImplementedError


class MSE(Loss):
    def forward(self, y, y_hat):
        return np.mean(np.power(y - y_hat, 2.0))

    def grad(self, y, y_hat):
        return -2 * (y - y_hat)
