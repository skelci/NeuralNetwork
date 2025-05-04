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
                2 * (target - neurons[-1]) * self.layers[-1].activation_derivative(neurons[-1]),
                neurons[-2]
            )
            new_weights[-1] += weight_change
            prev_layer_cost = 2 * (target - neurons[-1]) * self.layers[-1].activation_derivative(neurons[-1]) 
            new_biases[-1] += prev_layer_cost * learning_step

            for i in range(len(self.layers) - 2, -1, -1):
                current_cost = np.dot(prev_layer_cost, self.layers[i + 1].weights.T)
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

