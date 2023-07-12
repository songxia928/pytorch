
import numpy as np
import glob
import math
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import sys
sys.path.append('/media/mingyanzhen/project/utils_myz')
import file.rw_json as rw_json

class onePerson_3d(Dataset):
    def __init__(self, data_dir, info_file, data_size):
        self.data_dir = data_dir
        self.info_file = info_file
    
        self.data_size = data_size
        self.input_size = (data_size['n_slice'], data_size['h'], data_size['w'], data_size['c'])
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
        n_slice, _, _, _ = img.shape
        img_new = np.zeros(self.input_size)
        img_new[:n_slice, :, :, :] = img
        img_new = img_new.astype( np.float32 )
        img_new = np.swapaxes(np.swapaxes(img_new, 2, 3), 1, 2)
        #img_new = np.swapaxes(img_new, 0, 1)
        #video = [img_new[i] for i in range(self.input_size[0])]

        '''
        #transforms = self.transform()
        #tensor = transforms(img_new)
        totensor = transforms.ToTensor()
        norm = transforms.Normalize(self.mean, self.std)
        tensor = totensor(img_new)
        #tensor = norm(tensor)
        '''
        tensor = torch.from_numpy(img_new)
        label = self.filename2label[filename]

        return tensor, label

    def read_json(self, path_json):
        dic = rw_json.read_json(path_json)
        keys1, keys2 = ['T2', 'ADC'], ['origin', 'direction', 'space'] 
        vals = []
        for key1 in keys1:
             for key2 in keys2:
                 data = np.array(dic[key1][key2])
                 vals.append(data)
        emb = np.concatenate(vals, axis=0)
        return emb

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



