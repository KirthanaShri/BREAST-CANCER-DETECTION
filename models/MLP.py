import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, h: int, w: int, h_dim_list: list = [20, 10]):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        
        layer_dims = [h * w * 3] + h_dim_list + [2]
        ## e.g., layer_dims = [h * w * 3, 20, 10, 2]

        layer_list = []
        for i in range(len(layer_dims) - 1):
            new_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            layer_list.append(new_layer)
            if i != len(layer_dims) - 2:
                layer_list.append(nn.ReLU())
        # layers = [linear, relu, linear, relu, linear]
        self.layers = nn.Sequential(*layer_list)
        
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(h * w * 3, 20)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(20, 10)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(10, 2)  # Change to 1 for binary classification

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        # x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.fc3(x)
        return x
