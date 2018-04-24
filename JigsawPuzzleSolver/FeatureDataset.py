from numpy import sort
from torch.utils.data import Dataset
import torch
import os


class FeatureDataset(Dataset):
    def __init__(self, dir, file_list):
        self.dir = dir
        self.file_list = file_list

    def __getitem__(self, index):
        data = torch.load(self.dir + '/' + str(index))
        x = data[:, : -1]
        y = data[:, -1].long()
        return x, y

    def __len__(self):
        return len(self.file_list)

class FeatureDatasetGenerator:
    def __init__(self, dir):
        self.dir = dir
        self.file_list = [file for file in os.listdir(dir)]
        sort(self.file_list)

    def generate(self, ratio=0.2):
        data_len = len(self.file_list)
        train_dataset = FeatureDataset(self.dir, self.file_list[ : int(ratio * data_len)])
        val_dataset = FeatureDataset(self.dir, self.file_list[int(ratio * data_len) : ])

        return train_dataset, val_dataset