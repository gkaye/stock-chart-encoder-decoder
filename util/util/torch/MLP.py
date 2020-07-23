import torch
from util.torch.HiddenLayers import HiddenLayers


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_hidden_layers, dropout):
        super(MLP, self).__init__()

        # Make layers
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.input_layer = torch.nn.Linear(input_size, hidden_layer_size)
        self.hidden_layers = HiddenLayers(hidden_layer_size, num_hidden_layers, dropout)
        self.output_layer = torch.nn.Linear(hidden_layer_size, output_size)

    def forward(self, input):
        # Flatten anything after batch dimension
        flattened_input = input.flatten(start_dim=1)

        input_ret = self.dropout_layer(self.input_layer(flattened_input))
        hidden_ret = self.dropout_layer(self.hidden_layers(input_ret))
        ret = self.output_layer(hidden_ret)

        return ret
