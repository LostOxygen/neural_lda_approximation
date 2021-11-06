"""custom TensorDataset library"""
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    """custom TensorDataset class which inherits from Pytorchs Dataset class
       and applies a specified transform to all dataset items.
    """
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        data, targ = tuple(tensor[index] for tensor in self.tensors)
        return data, targ

    def __len__(self):
        return self.tensors[0].size(0)
