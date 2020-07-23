import torch
from util.torch.HiddenLayer import HiddenLayer


class HiddenLayers(torch.nn.Module):
    def __init__(self, layer_input_size, num_layers, dropout):
        super().__init__()

        self.layers = []

        for i in range(num_layers):
            is_last = i + 1 == num_layers

            # Don't pass dropout for last layer
            self.layers.append(HiddenLayer(layer_input_size, 0. if is_last else dropout))

        # Needs to be a 'ModuleList' to allow pytorch internal controls
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
