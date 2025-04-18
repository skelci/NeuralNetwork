import numpy as np
from enum import IntEnum



class LayerType(IntEnum):
    Sigmoid = 0
    ReLU = 1
    LeakyReLU = 2
    ELU = 3
    Tanh = 4
    BinaryStep = 5
    Raw = 6



class Layer:
    def __init__(self, size, type, alpha=None):
        self.size = size
        self.type = type
        self.alpha = alpha

        self.activation_function = {
            LayerType.Sigmoid:      lambda x: 0.5 * (1.0 + np.tanh(x * 0.5)),
            LayerType.ReLU:         lambda x: np.maximum(0, x),
            LayerType.LeakyReLU:    lambda x, alpha=0.01: np.where(x > 0, x, alpha * x),
            LayerType.ELU:          lambda x, alpha=1.0: np.where(x > 0, x, alpha * (np.exp(x) - 1)),
            LayerType.Tanh:         lambda x: np.tanh(x),
            LayerType.BinaryStep:   lambda x: np.where(x >= 0, 1, 0),
            LayerType.Raw:          lambda x: x
        }[type]

        self.activation_derivative = {
            LayerType.Sigmoid:      lambda x: x * (1 - x),
            LayerType.ReLU:         lambda x: np.where(x > 0, 1, 0),
            LayerType.LeakyReLU:    lambda x, alpha=0.01: np.where(x > 0, 1, alpha),
            LayerType.ELU:          lambda x, alpha=1.0: np.where(x > 0, 1, alpha * np.exp(x)),
            LayerType.Tanh:         lambda x: 1 - np.square(x),
            LayerType.BinaryStep:   lambda x: np.zeros_like(x),
            LayerType.Raw:          lambda x: np.ones_like(x)
        }[type]


    def load(self, input_size, weights=None, biases=None):
        self.input_size = input_size

        if weights is None:
            self.weights = np.random.normal(loc=0, scale=np.sqrt(1/self.input_size), size=(self.size, input_size))
        else:
            if weights.shape != (self.size, input_size):
                raise ValueError(f"Invalid weights shape: {weights.shape}, expected: {(self.size, input_size)}")
            self.weights = weights

        if biases is None:
            self.biases = np.zeros(self.size)
        else:
            if biases.shape != (self.size,):
                raise ValueError(f"Invalid biases shape: {biases.shape}, expected: {(self.size,)}")
            self.biases = biases


    def forward(self, input):
        if input.shape != (self.input_size,):
            raise ValueError(f"Invalid input shape: {input.shape}, expected: {(self.input_size,)}")
        step_1 = np.dot(self.weights, input) + self.biases
        step_2 = self.activation_function(step_1) if self.alpha is None else self.activation_function(step_1, self.alpha)
        return step_2

