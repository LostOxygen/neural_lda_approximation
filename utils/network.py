"""library module for the used DNN model"""
import torch
import torch.nn as nn

class DNN(nn.Module):
    """a standard CNN model which performs well on the CIFAR10 dataset"""
    def __init__(self, num_topics: int, input_dim: int) -> None:
        super(DNN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, num_topics*3),
            nn.Tanh(),
            nn.Linear(num_topics*3, num_topics*2),
            nn.Tanh(),
            nn.Linear(num_topics*2, num_topics),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward function of the model"""
        x = torch.flatten(x, 1) #Flatten
        x = self.fc(x)
        return x
