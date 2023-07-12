
import numpy as np
import glob
import math
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class Slice(Dataset):
    def __init__(self, data_dir, info_file):
        self.data_dir = data_dir
        self.info_file = info_file
        self.input_size = (160, 160, 2)
        self.mean = [0.5, 0.5]
        self.std = [0.5, 0.5]
        self.input_range = [0, 1]
        self.scale = 0.875
        self.space = 'RGB'
        self.filenames, self.filename2label = self.read_info()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path_file = self.data_dir + '/' + filename
        img = np.load(path_file)
        img = img.astype( np.float32 )
        #img = Image.fromarray(array)
        transforms = self.transform()
        tensor = transforms(img)
        label = self.filename2label[filename]
        return tensor, label

    def read_info(self):
        with open(self.info_file, 'r') as fr:
            lines = fr.readlines()
        datas = [line.strip().split('\t') for line in lines]
        filenames = [data[0] for data in datas]
        filename2label = {data[0]:int(data[1]) for data in datas}
        return filenames, filename2label

    def transform(self):
        tfs = []
        #tfs.append(transforms.ToPILImage())
        #tfs.append(transforms.Resize(int(math.floor(max(self.input_size) / self.scale))))
        #tfs.append(transforms.CenterCrop(max(self.input_size)))
        tfs.append(transforms.ToTensor())
        tfs.append(transforms.Normalize(self.mean, self.std))
        tf = transforms.Compose(tfs)
        return tf



