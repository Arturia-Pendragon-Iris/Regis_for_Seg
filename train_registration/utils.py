import glob
import os
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
from monai.transforms import *

target_size = (512, 512)
# target_size = (256, 256, 192)
train_transforms = Compose([
    RandAffine(
        prob=0.5,
        padding_mode="zeros",
        spatial_size=target_size,
        translate_range=(32, 32),
        rotate_range=(np.pi / 10, np.pi / 10),
        scale_range=(-0.1, 0.1)),
    RandGridDistortion(
        padding_mode="zeros",
        mode="nearest",
        prob=0.5),
    RandGaussianSharpen(prob=0.1),
    RandGaussianSmooth(prob=0.1),
    # RandHistogramShift(num_control_points=10, prob=0.2),
    RandAxisFlip(prob=0.5),
    RandRotate90(prob=0.2, spatial_axes=(0, 1)),
    Resize(target_size)
]
)


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.file_list = glob.glob((os.path.join(dataset_dir, "*.npz")))

    def __getitem__(self, index):
        # print(self.file_list[index])
        ct_1 = np.load(os.path.join(self.dataset_dir, self.file_list[index]))["arr_0"]

        index_2 = np.random.randint(low=0, high=len(self.file_list))
        ct_2 = np.load(os.path.join(self.dataset_dir, self.file_list[index_2]))["arr_0"]

        ct = np.stack((ct_1, ct_2), axis=0)
        ct = train_transforms(ct)
        ct = torch.tensor(ct).to(torch.float).to("cuda")

        return ct

    def __len__(self):
        return len(self.file_list)
