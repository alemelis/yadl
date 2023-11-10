from dataclasses import dataclass
from typing import List

from yadl.layer import Layer


@dataclass
class Network:
    name: str = ""
    layers: List[Layer] = None

    def randomize(self):
        for layer in self.layers:
            layer.randomize()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad, lr):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, lr)
        return loss_grad

    def __str__(self):
        s = f"{self.name}\n"
        s += "-" * len(self.name) + "\n"
        s += "\n".join(str(l) for l in self.layers)
        return s

    def summary(self):
        print(self)
