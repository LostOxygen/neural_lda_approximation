"""library module for the used DNN model"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    """a standard CNN model which performs well on the CIFAR10 dataset"""
    def __init__(self, num_topics: int, input_dim: int) -> None:
        super().__init__()

        self.fully_connected = nn.Sequential(
            nn.Linear(input_dim, num_topics*4),
            nn.Tanh(),
            nn.Linear(num_topics*4, num_topics*3),
            nn.Tanh(),
            nn.Linear(num_topics*3, num_topics*2),
            nn.Tanh(),
            nn.Linear(num_topics*2, num_topics)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward function of the model"""
        #x = torch.flatten(x, 1) #Flatten
        x = self.fully_connected(x)
        return x


class CustomCrossEntropy(torch.nn.Module):
    """https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720
    """
    def __init__(self):
        super().__init__()

    def forward(self, prediction: float, target: float) -> float:
        """forward method to calculate the loss for a given prediction and soft_targets"""
        log_probs = F.log_softmax(prediction, dim=-1)
        return torch.mean(torch.sum(-target * log_probs, -1))
