"""NumPy 前馈神经网络：等价于原 torch 网络（float32），去掉框架开销，支持单条与批量。"""
import numpy as np

DTYPE = np.float32


def _sigmoid(x):
    """Numerically stable sigmoid (matches torch's overflow-safe behavior)."""
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


class Net:
    """A small MLP implemented directly with NumPy.

    Attributes:
        a: Size of input layer.
        b: Size of hidden layer 1.
        c: Size of hidden layer 2.
        d: Size of output layer.
        W1, W2, W3: Weight matrices.
        b1, b2, b3: Bias vectors.
    """

    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, weights):
        self.a = n_input
        self.b = n_hidden1
        self.c = n_hidden2
        self.d = n_output
        self.update_weights(weights)

    def update_weights(self, weights):
        """Unpack a flat gene vector into weight matrices and biases.

        weights has size a*b+b + b*c+c + c*d+d.
        """
        weights = np.asarray(weights, dtype=DTYPE)
        x = self.a * self.b
        xx = x + self.b
        y = xx + self.b * self.c
        yy = y + self.c
        z = yy + self.c * self.d

        self.W1 = weights[0:x].reshape(self.b, self.a)
        self.b1 = weights[x:xx]
        self.W2 = weights[xx:y].reshape(self.c, self.b)
        self.b2 = weights[y:yy]
        self.W3 = weights[yy:z].reshape(self.d, self.c)
        self.b3 = weights[z:]

    def forward(self, x):
        """Forward pass. Accepts a 1-D sample or a 2-D batch; sigmoid output."""
        x = np.asarray(x, dtype=DTYPE)
        h1 = np.maximum(0.0, x @ self.W1.T + self.b1)
        h2 = np.maximum(0.0, np.asarray(h1, dtype=DTYPE) @ self.W2.T + self.b2)
        logits = np.asarray(h2, dtype=DTYPE) @ self.W3.T + self.b3
        return _sigmoid(np.asarray(logits, dtype=DTYPE))

    def predict(self, state):
        """Return the action (argmax) for a single state vector."""
        return int(self.forward(np.asarray(state, dtype=DTYPE)).argmax())


if __name__ == "__main__":
    import random

    from settings import N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT, GENES_LEN

    weights = [random.uniform(-1, 1) for _ in range(GENES_LEN)]
    model = Net(N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT, weights)
    sample = [random.random() for _ in range(N_INPUT)]
    print("single:", model.predict(sample))
    batch = np.random.random((5, N_INPUT))
    print("batch argmax:", model.forward(batch).argmax(axis=1))
