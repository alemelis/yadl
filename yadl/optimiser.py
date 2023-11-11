import random
from dataclasses import dataclass

from tqdm import tqdm

from yadl.loss import Loss
from yadl.network import Network


@dataclass
class SGD:
    net: Network
    loss_func: Loss

    def __post_init__(self):
        self.loss = self.loss_func()
        self.loss_log = []
        self.test_loss_log = []

    def fit(
        self,
        train_x,
        train_y,
        epochs=10,
        lr=1e-3,
        test_x=None,
        test_y=None,
        initialise=True,
    ):
        if initialise:
            self.net.randomize()

        num_samples = len(train_x)
        dataset = list(zip(train_x, train_y))

        for _ in tqdm(
            range(epochs),
            desc="Training",
            unit=" epochs",
        ):
            random.shuffle(dataset)  # Stochastic

            loss = 0.0
            for x, y in dataset:
                # forward
                y_hat = self.net.forward(x)
                loss += self.loss.forward(y, y_hat)

                # backward
                loss_grad = self.loss.grad(y, y_hat)
                loss_grad = self.net.backward(loss_grad, lr)  # Gradient Descent

            loss /= num_samples
            self.loss_log.append(loss)

            test_loss = 0.0
            if test_x is not None:
                for x, y in zip(test_x, test_y):
                    y_hat = self.net.forward(x)
                    test_loss += self.loss.forward(y, y_hat)
                test_loss /= len(test_x)
                self.test_loss_log.append(test_loss)

        print(f"\nloss = {loss:4.3e}")
        if test_x is not None:
            print(f"test_loss = {test_loss:4.3e}")
