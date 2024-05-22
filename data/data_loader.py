from typing import Callable

import pandas as pd
import torch
import torch.utils.data as data

from configs.config import DatasetConfig, Mode


class MITLoader_MLP(data.Dataset):

    def __init__(self, csv_file, transforms: Callable = lambda x: x) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file).values
        self.transforms = transforms

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, item):
        signal = self.annotations[item, :-1]
        label = int(self.annotations[item, -1])
        # TODO: add augmentations
        signal = torch.from_numpy(signal).float()
        signal = self.transforms(signal)

        return signal, torch.tensor(label).long()
    
    
    
class MITLoader_CNN_Transformer(data.Dataset):

    def __init__(self, csv_file, transforms: Callable = lambda x: x) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file).values
        self.transforms = transforms

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, item):
        signal = self.annotations[item, :-1]
        label = int(self.annotations[item, -1])
        # TODO: add augmentations
        signal = torch.from_numpy(signal).float()
        signal = self.transforms(signal)

        return signal.unsqueeze(0), torch.tensor(label).long()



