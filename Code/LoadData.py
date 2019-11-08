import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt


class DatasetLoader(Dataset):
    def __init__(self,folder_path,source,target):
        self.folder_path = folder_path
        self.hr_folder = self.folder_path+target
        self.lr_folder = self.folder_path+source
        self.files_list = os.listdir(self.hrFolder)
        self.no_of_files = len(self.files_list)
    def __getitem__(self, id):
        file_name = self.files_list[id]
        hr_img = plt.imread(self.hr_folder + "/" + file_name)
        lr_img = plt.imread(self.lr_folder + "/" + file_name)
        return hr_img,lr_img
    def __len__(self):
        return self.no_of_files
