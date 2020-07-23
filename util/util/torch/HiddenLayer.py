import torch
import torch.nn.functional as F


class HiddenLayer(torch.nn.Module):
    def __init__(self, layer_input_size, dropout=0.):
        super().__init__()

        self.dropout_layer = torch.nn.Dropout(dropout)
        self.layer = torch.nn.Linear(layer_input_size, layer_input_size)
        self.norm = torch.nn.BatchNorm1d(layer_input_size)

    def forward(self, input):
        ret = self.layer(input)
        ret = self.norm(ret)
        ret = F.relu(ret)
        ret = self.dropout_layer(ret)
        return ret
