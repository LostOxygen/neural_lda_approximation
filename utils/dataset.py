"""custom TensorDataset library"""
import json
import torch
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


class WikiDataset(Dataset):
    """custom WikiDataset class which inherits from Pytorchs Dataset class
       and provides the wikipedia corpus as a dataset through a gensim lda model.
    """
    def __init__(self, file_list: list, label_list: list, dict_len: int):
        self.file_list = file_list
        self.label_list = label_list
        self.dict_len = dict_len
        assert len(self.file_list) == len(self.label_list)

    def __getitem__(self, index):
        data = []
        targets = []

        with open(self.file_list[index]) as file:
            dump = json.load(file)
            empty = torch.zeros(self.dict_len)
            for key, val in dump.items():
                empty[int(key)] = float(val)
            data.append(empty)

        with open(self.label_list[index]) as file:
            # read the line and sanitize the string and convert it back to an int list
            tmp_str = file.readlines()
            tmp_str = list(map(float, tmp_str[0].replace("[", "").replace("]", "").split(",")))
            targets.append(tmp_str)

        data = torch.FloatTensor(data)
        targets = torch.FloatTensor(targets)
        return data, targets


    def __len__(self):
        return len(self.file_list)
