from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the Linear_QNet class.

        Parameters:
        input_size (int): The input size.
        hidden_size (int): The hidden size.
        output_size (int): The output size.
        """

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


class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        """
        Initialize the QTrainer class.

        Parameters:
        model (pytorch model): The model to train.
        lr (float): The learning rate.
        gamma (float): The discount rate.
        """
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Performs a training step.

        Parameters:
        state (torch.tensor): The current state.
        action (torch.tensor): The action taken.
        reward (torch.tensor): The reward received.
        next_state (torch.tensor): The next state.
        done (bool): Whether the episode is done or not.
        """

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # Add batch dimension for single step
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()

        # 2: Iterate over the batch
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx]),
                )

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3: Backpropagation
        self.optimizer.zero_grad()  # Reset the gradients
        loss = self.criterion(target, pred)  # Calculate the loss
        loss.backward()  # Perform backpropagation
        self.optimizer.step()  # Update the weights
