import torchvision.transforms as torch_transform

import numpy as np
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def hr_transform(crop_size = 96):
    return torch_transform.Compose([
        torch_transform.ToPILImage(),
        torch_transform.RandomCrop(crop_size),
        torch_transform.ToTensor()
    ])

def lr_transform(crop_size = 96, upscale_factor = 4):
    return torch_transform.Compose([
        torch_transform.ToPILImage(),
        torch_transform.Resize(crop_size//upscale_factor,interpolation=Image.BICUBIC),
        torch_transform.ToTensor()
    ])

class Dataset_Train(Dataset):
    def __init__(self, dirpath, crop_size = 96, upscale_factor = 4):
        super(Dataset_Train, self).__init__()
        self.imagelist = glob.glob(os.path.join(dirpath,"*.jpg"))

        self.cropsize = crop_size - (crop_size%upscale_factor)

        self.hr_transform = hr_transform(self.cropsize)
        self.lr_transform = lr_transform(self.cropsize, upscale_factor=upscale_factor)

    def __getitem__(self, index):
        image = Image.open(self.imagelist[index])
        hr_image = self.hr_transform(image)
        lr_image = self.lr_transform(hr_image)

        return lr_image, hr_image

    def __len__(self):
        return len(self.imagelist)

class Dataset_Vaild(Dataset):
    def __init__(self, dirpath, upscale_factor = 4):
        super(Dataset_Vaild, self).__init__()
        self.upscale_factor = upscale_factor
        self.imagelist = glob.glob(os.path.join(dirpath,"*.jpg"))

    def __getitem__(self, index):
        image = Image.open(self.imagelist[index])
        height, width = image.size
        self.crop_size = min(height,width) - (min(height,width) % self.upscale_factor)
        self.hr_transform = hr_transform(self.crop_size)
        self.lr_transform = lr_transform(self.crop_size, self.upscale_factor)

        hr_image = self.hr_transform(self.image)
        lr_image = self.lr_transform(hr_image)

        return lr_image, hr_image

    def __len__(self):
        return len(self.imagelist)


if __name__ == "__main__":
    dirpath_train = "Data/Train"
    dirpath_vaild = "Data/Vaild"
    Test_Traindataset = Dataset_Train(dirpath= dirpath_train, crop_size= 96, upscale_factor= 4)
    Test_Vailddataset = Dataset_Vaild(dirpath= dirpath_vaild, upscale_factor=4)

    Test_TraindataLoader = DataLoader(dataset=Test_Traindataset, batch_size=16, shuffle=True)
    Test_VailddataLoader = DataLoader(dataset=Test_Vailddataset, batch_size=1)

    for i,(input, target) in enumerate(Test_TraindataLoader):
        plt.imshow(input[0])
        plt.show()
        plt.imshow(target[0])
        plt.show()

