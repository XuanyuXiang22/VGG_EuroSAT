import os
import torch
import utils
import numpy as np
from torch.utils.data import DataLoader, Dataset


class VGGDataset(Dataset):
    def __init__(self,
                 dataroot,
                 is_train,
                 RandomResizedCrop_p,
                 RandomResizedCrop_scale):

        self.is_train = is_train
        self.data_paths = utils.make_paths(dataroot)  # ['dataroot/AnnualCrop/AnnualCrop_1.tif', ...]
        self.pipeline = utils.get_pipeline(RandomResizedCrop_p, RandomResizedCrop_scale, self.is_train)

        self.cls_corr = {
            "AnnualCrop": 0,
            "Forest": 1,
            "HerbaceousVegetation": 2,
            "Highway": 3,
            "Industrial": 4,
            "Pasture": 5,
            "PermanentCrop": 6,
            "Residential": 7,
            "River": 8,
            "SeaLake": 9
        }

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]  # 'dataroot/AnnualCrop/AnnualCrop_1.tif'
        img = utils.get_rawData(data_path)  # [13, 64, 64] np.array float32 ~ [0, 10000]
        img = np.transpose(img, (1, 2, 0))  # [64, 64, 13], this is the right input size for transforms.ToTensor

        # get label
        cls = os.path.basename(data_path).split('_')[0]  # AnnualCrop
        label = self.cls_corr[cls]  # 0
        label = torch.tensor(label, dtype=torch.long)
        # transform img
        img = self.pipeline(img)

        return img, label


def get_dataloader(dataroot, batch_size, isTrain, RandomResizedCrop_p, RandomResizedCrop_scale):
    dataset = VGGDataset(dataroot, isTrain, RandomResizedCrop_p, RandomResizedCrop_scale)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=isTrain,
        drop_last=isTrain
    )
    return dataloader