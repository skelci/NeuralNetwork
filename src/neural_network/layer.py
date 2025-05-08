import numpy as np
from enum import IntEnum



class LayerType(IntEnum):
    Sigmoid = 0
    ReLU = 1
    LeakyReLU = 2
    ELU = 3
    Tanh = 4
    BinaryStep = 5
    Linear = 6



class Layer:
    def __init__(self, type, size, alpha=None):
        self.size = size
        self.type = type
        self.alpha = alpha

        # numerically stable sigmoid
        def sigmoid(z):
            return 0.5 * (1.0 + np.tanh(0.5 * z))

        self.activation_function = {
            LayerType.Sigmoid:      sigmoid,
            LayerType.ReLU:         lambda z: np.maximum(0, z),
            LayerType.LeakyReLU:    lambda z, alpha=0.01: np.where(z > 0, z, alpha * z),
            LayerType.ELU:          lambda z, alpha=1.0: np.where(z > 0, z, alpha * (np.exp(z) - 1)),
            LayerType.Tanh:         lambda z: np.tanh(z),
            LayerType.BinaryStep:   lambda z: np.where(z >= 0, 1, 0),
            LayerType.Linear:       lambda z: z
        }[type]

        self.activation_derivative = {
            LayerType.Sigmoid:      lambda z: sigmoid(z) * (1 - sigmoid(z)),
            LayerType.ReLU:         lambda z: np.where(z > 0, 1, 0),
            LayerType.LeakyReLU:    lambda z, alpha=0.01: np.where(z > 0, 1, alpha),
            LayerType.ELU:          lambda z, alpha=1.0: np.where(z > 0, 1, alpha * np.exp(z)),
            LayerType.Tanh:         lambda z: 1 - np.tanh(z)**2,
            LayerType.BinaryStep:   lambda z: np.zeros_like(z),
            LayerType.Linear:       lambda z: np.ones_like(z)
        }[type]


    def get_activation_derivative(self):
        if self.alpha is None:
            return self.activation_derivative(self.last_z)
        else:
            return self.activation_derivative(self.last_z, self.alpha)


    def load(self, input_size, weights=None, biases=None):
        self.input_size = input_size

        if weights is None:
            self.weights = np.random.normal(loc=0, scale=np.sqrt(1/self.input_size), size=(self.size, input_size))
            self.weights = self.weights.astype(np.float16)
        else:
            if weights.shape != (self.size, input_size):
                raise ValueError(f"Invalid weights shape: {weights.shape}, expected: {(self.size, input_size)}")
            self.weights = weights

        if biases is None:
            self.biases = np.zeros(self.size, dtype=np.float16)
        else:
            if biases.shape != (self.size,):
                raise ValueError(f"Invalid biases shape: {biases.shape}, expected: {(self.size,)}")
            self.biases = biases


    def forward(self, input):
        if input.shape != (self.input_size,):
            raise ValueError(f"Invalid input shape: {input.shape}, expected: {(self.input_size,)}")
        z = np.dot(self.weights, input) + self.biases
        self.last_z = z
        a = (self.activation_function(z)
            if self.alpha is None
            else self.activation_function(z, self.alpha)
        )
        return a

