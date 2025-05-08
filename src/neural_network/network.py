from .layer import *

import os
import numpy as np



class Network:
    def save(self, path):
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)

        with open(path, "wb") as f:
            f.write(np.uint8(len(self.layers)).tobytes())
            f.write(np.uint16(self.input_size).tobytes())
            for layer in self.layers:
                if layer.alpha is None:
                    alpha = -1
                else:
                    alpha = layer.alpha
                f.write(np.uint16(layer.size).tobytes())
                f.write(np.float16(alpha).tobytes())
                f.write(np.uint8(layer.type).tobytes())
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
            num_layers = np.frombuffer(f.read(1), dtype=np.uint8)[0]
            prev_size = np.frombuffer(f.read(2), dtype=np.uint16)[0]
            self.input_size = prev_size
            self.layers = []
            for _ in range(num_layers):
                size = np.frombuffer(f.read(2), dtype=np.uint16)[0]
                alpha = np.frombuffer(f.read(2), dtype=np.float16)[0]
                layer_type = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                weights = np.frombuffer(f.read(size * prev_size * 2), dtype=np.float16).reshape((size, prev_size))
                biases = np.frombuffer(f.read(size * 2), dtype=np.float16)

                layer = Layer(
                    LayerType(layer_type),
                    size,
                    alpha if alpha != -1 else None,
                )
                layer.load(prev_size, weights.copy(), biases.copy())
                self.layers.append(layer)
                prev_size = size


    def get_result(self, input):
        if input.shape != (self.layers[0].input_size,):
            raise ValueError(f"Invalid input shape: {input.shape}, expected: {(self.layers[0].input_size,)}")
        for layer in self.layers:
            input = layer.forward(input)
        return input
    

    def get_cost(self, data):
        cost = 0.0
        for input, target in data:
            output = self.get_result(input)
            cost += np.sum(np.square(target - output))
        return cost / len(data)
    

    def back_prop(self, data, learning_step=0.01):
        new_weights = [np.zeros_like(layer.weights) for layer in self.layers]
        new_biases = [np.zeros_like(layer.biases) for layer in self.layers]

        for input, target in data:
            neurons = [np.array(input, dtype=np.float16)]

            for layer in self.layers:
                neurons.append(layer.forward(neurons[-1]))

            weight_change = learning_step * np.outer(
                (target - neurons[-1]) * 2 * self.layers[-1].activation_derivative(neurons[-1]),
                neurons[-2]
            )
            new_weights[-1] += weight_change
            prev_layer_cost = (target - neurons[-1]) * 2 * self.layers[-1].activation_derivative(neurons[-1]) 
            new_biases[-1] += prev_layer_cost * learning_step

            for i in range(len(self.layers) - 2, -1, -1):
                current_cost = np.dot(prev_layer_cost, self.layers[i + 1].weights)
                weight_change = learning_step * np.outer(
                    current_cost * self.layers[i].activation_derivative(neurons[i + 1]),
                    neurons[i]
                )
                prev_layer_cost = current_cost
                new_weights[i] += weight_change
                new_biases[i] += current_cost * self.layers[i].activation_derivative(neurons[i + 1]) * learning_step

        new_weights = [w / len(data) for w in new_weights]
        new_biases = [b / len(data) for b in new_biases]

        for i in range(len(self.layers)):
            self.layers[i].weights += new_weights[i]
            self.layers[i].biases += new_biases[i]

