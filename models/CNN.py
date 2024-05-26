import torch
import torch.nn as nn
from typing import List

class MyCNN(nn.Module): 
    def __init__(
        self,
        num_kernel_conv_list: List[int],
        use_pooling: bool,
        kernel_size: int = 3,
        input_dim: int = 3
    ):
        super().__init__() 
        self.conv1 = nn.Conv2d(
            in_channels=3,  ### number of channels of original images
            out_channels= num_kernel_conv_list[0],
            kernel_size=(kernel_size, kernel_size),
            stride=1,
            padding=1
        )
        conv_layer_list = []
        
        for i in range(len(num_kernel_conv_list)-1):
            new_conv_layer = nn.Conv2d(
                in_channels=num_kernel_conv_list[i],
                out_channels=num_kernel_conv_list[i+1],
                kernel_size=(kernel_size, kernel_size),
                stride=1,
                padding=1,
            )
            conv_layer_list.append(new_conv_layer)
            conv_layer_list.append(nn.ReLU())
            if use_pooling:
                conv_layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_layers = nn.Sequential(*conv_layer_list)
    
        output_size = self._get_output_size()
        self.linear = nn.Linear(
            in_features = num_kernel_conv_list[-1] * output_size * output_size, out_features=2)

    @torch.no_grad()
    def _get_output_size(self):
        device = next(self.parameters()).device
        dummy_input = torch.zeros(1, 3, 224, 224).to(device)
        dummy_output_1 = self.conv1(dummy_input)
        dummy_output = self.conv_layers(dummy_output_1)
        return dummy_output.shape[2]
    
    def forward(self, x: torch.Tensor):
        ## x -> conv1 -> relu_1 -> conv2 -> relu_2 -> conv3 -> relu_3 ->(flatten) -> linear -> softmax
        # x = self.conv_1(x)
        # x = self.relu_1(x)
        # x = self.conv_2(x)
        # x = self.relu_2(x)
        # x = self.conv_3(x)
        # x = self.relu_3(x)
        # print mean and std of x per channel
        # print("mean of x per channel: ", x.mean(dim=(0, 2, 3)))
        # print("std of x per channel: ", x.std(dim=(0, 2, 3)))

        out = self.conv1(x)
        out = self.conv_layers(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)

        return out
        # b = x.shape[0]
        # x = x.reshape(b, -1)  ## flatten
        # x = self.linear(x)

        ## return output
        return x



        # self.conv_1 = nn.Conv2d(
        #     in_channels=3,  ### number of channels of original images
        #     out_channels=num_kernel_conv1,
        #     kernel_size=(kernel_size, kernel_size),
        #     stride=1,
        #     padding=1,
        # )
        # self.conv_2 = nn.Conv2d(
        #     in_channels=num_kernel_conv1,
        #     out_channels=num_kernel_conv2,
        #     kernel_size=(kernel_size, kernel_size),
        #     stride=1,
        #     padding=1,
        # )
        # self.conv_3 = nn.Conv2d(
        #     in_channels=num_kernel_conv2,
        #     out_channels=num_kernel_conv3,
        #     kernel_size=(kernel_size, kernel_size),
        #     stride=1,
        #     padding=1,
        # )
       