from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters:
        x (torch.tensor): The input tensor.

        Returns:
        x (torch.tensor): The output tensor.
        """

        x = F.relu(self.linear1(x))  # Apply relu to the first layer
        x = self.linear2(x)

        return x

    def save_to_file(self, file_name='model.pth'):
        """
        Saves the model to a file.

        Parameters:
        file_name (str): The name of the file. Defaults to 'model.pth'.
        """

        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
