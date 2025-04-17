from .layer import *

import os
import numpy as np



class Network:
    def save(self, path):
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)

        with open(path, "wb") as f:
            f.write(len(self.layers).to_bytes(1))
            f.write(np.array([self.input_size] + [layer.size for layer in self.layers], dtype=np.int32).tobytes())
            f.write(np.array([layer.type for layer in self.layers], dtype=np.int8).tobytes())
            f.write(np.array([layer.alpha if layer.alpha else 0 for layer in self.layers], dtype=np.float16).tobytes())
            for layer in self.layers:
                f.write(layer.weights.tobytes())
                f.write(layer.biases.tobytes())


    def create(self, input_size, layers):
        self.input_size = input_size
        self.layers = layers
        prev_size = self.input_size
        for layer in self.layers:
            layer.load(prev_size)
            prev_size = layer.size


    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(path, "rb") as f:
            num_layers = int.from_bytes(f.read(1))
            sizes = np.frombuffer(f.read(4 * (num_layers + 1)), dtype=np.int32)
            types = np.frombuffer(f.read(num_layers), dtype=np.int8)
            alphas = np.frombuffer(f.read(num_layers * 2), dtype=np.float16)
            self.input_size = sizes[0]
            self.layers = []
            for i in range(num_layers):
                self.layers.append(Layer(
                    sizes[i + 1],
                    LayerType(types[i]),
                    alphas[i] if alphas[i] != 0 else None
                ))

            prev_size = self.input_size
            for layer in self.layers:
                l_size = layer.size
                weights = np.frombuffer(f.read(l_size * prev_size * 4), dtype=np.float32).reshape((l_size, prev_size))
                biases = np.frombuffer(f.read(l_size * 4), dtype=np.float32)
                layer.load(prev_size, weights, biases)
                prev_size = layer.size


    def get_result(self, input):
        if input.shape != (self.layers[0].input_size,):
            raise ValueError(f"Invalid input shape: {input.shape}, expected: {(self.layers[0].input_size,)}")
        for layer in self.layers:
            input = layer.forward(input)
        self.prev_result = input
        return input
    

    def get_cost(self, target):
        if self.prev_result is None:
            raise ValueError("No previous result found. Please call get_result() first.")
        if target.shape != self.prev_result.shape:
            raise ValueError(f"Invalid target shape: {target.shape}, expected: {self.prev_result.shape}")
        return np.sum((self.prev_result - target) ** 2)
