import torchvision.transforms as torch_transform
import torch
import numpy as np
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import utils

def hr_transform(crop_size = 96):
    transform = torch_transform.Compose([
        torch_transform.ToPILImage(),
        torch_transform.RandomCrop(crop_size), #raise (ValueError : empty range for randrange())
       # torch_transform.CenterCrop(crop_size),
        torch_transform.ToTensor()
    ])
    return transform

def lr_transform(crop_size = 96, upscale_factor = 4):
    transform = torch_transform.Compose([
        torch_transform.ToPILImage(),
        torch_transform.Resize(int(crop_size//upscale_factor),interpolation=Image.BICUBIC),
        torch_transform.ToTensor()
    ])
    return transform

class Dataset_Train(Dataset):
    def __init__(self, dirpath, crop_size = 96, upscale_factor = 4):
        super(Dataset_Train, self).__init__()
        self.imagelist = glob.glob(os.path.join(dirpath,"*.jpg"))
      # self.cropsize = crop_size - (crop_size%upscale_factor)
        self.cropsize = crop_size
        self.hr_transform = hr_transform(self.cropsize)
        self.lr_transform = lr_transform(self.cropsize, upscale_factor=upscale_factor)

    def __getitem__(self, index):
        image = Image.open(self.imagelist[index])
        image = np.array(image)
       # cropped_image = utils.randomcrop(image,self.cropsize)
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
        image = np.array(image)
        height, width = image.shape[0], image.shape[1]
        self.crop_size = min(height,width) - (min(height,width) % self.upscale_factor)
        self.hr_transform = hr_transform(self.crop_size)
        self.lr_transform = lr_transform(self.crop_size, self.upscale_factor)

        hr_image = self.hr_transform(image)
        lr_image = self.lr_transform(hr_image)
        print("size of hr_image : {}".format(hr_image.shape))
        print("size of hr_image : {}".format(lr_image.shape))
        return lr_image, hr_image

    def __len__(self):
        return len(self.imagelist)

class Dataset_Test(Dataset):
    def __init__(self, dirpath):
        super(Dataset_Test, self).__init__()
       # self.upscale_factor = upscale_factor
        self.imagelist = glob.glob(os.path.join(dirpath, "*.jpg"))
        self.transform = torch_transform.Compose([
            torch_transform.ToPILImage(),
            torch_transform.CenterCrop(96),
            torch_transform.ToTensor()
        ])

    def __getitem__(self, index):
        image = Image.open(self.imagelist[index])
        image = np.array(image)
        print("Test : image size {}".format(image.shape))
  #      image = np.transpose(image,(2,0,1))
        #image = torch.from_numpy(image)
        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.imagelist)


if __name__ == "__main__":
    dirpath_train = "Dataset/Train"
    dirpath_vaild = "Dataset/Vaild"
    Test_Traindataset = Dataset_Train(dirpath= dirpath_train, crop_size= 96, upscale_factor= 4)
    Test_Vailddataset = Dataset_Vaild(dirpath= dirpath_vaild, upscale_factor=4)

    Test_TraindataLoader = DataLoader(dataset=Test_Traindataset, batch_size=16, shuffle=True)
    Test_VailddataLoader = DataLoader(dataset=Test_Vailddataset, batch_size=1)

    for input, target in Test_TraindataLoader:
        print(input[0].shape)
        inputimage = np.array(input[0])
        print(inputimage)
        inputimage = np.transpose(inputimage,(1,2,0))
        plt.imshow(inputimage)
        plt.show()
        outputimage = np.array(target[0])
        outputimage = np.transpose(outputimage,(1,2,0))
        plt.imshow(outputimage)
        plt.show()
        break

    for input, target in Test_VailddataLoader:
        print(input[0].shape)
        inputimage = np.array(input[0])
        print(inputimage)
        inputimage = np.transpose(inputimage, (1, 2, 0))
        plt.imshow(inputimage)
        plt.show()
        outputimage = np.array(target[0])
        outputimage = np.transpose(outputimage, (1, 2, 0))
        plt.imshow(outputimage)
        plt.show()
        break
