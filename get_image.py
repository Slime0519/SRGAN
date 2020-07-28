import torch
import Model
import Perceptual_Loss
import Dataset_gen
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

DIRPATH_TRAIN = "Dataset/Train"
DIRPATH_VAILD = "Dataset/Vaild"

UPSCALE_FACTOR = 4

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":

    train_dataset = Dataset_gen.Dataset_Train(dirpath=DIRPATH_TRAIN, crop_size=96, upscale_factor=UPSCALE_FACTOR)
   # vaild_dataset = Dataset_gen.Dataset_Vaild(dirpath=DIRPATH_VAILD, upscale_factor=UPSCALE_FACTOR)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1,)

    Generator = Model.Generator().to(device)
    Generator.load_state_dict(torch.load("Trained_model/Generator/generator_299th_model.pth"))

    Vggloss = Perceptual_Loss.vggloss(truncate_layer=36).to(device)

    for input, target in train_dataloader:
        input_temp = input
        target_temp = target

        input, target = input.to(device), target.to(device)
        input_tempimage = np.array(input_temp).squeeze()
        target_tempimage = np.array(target_temp).squeeze()

        input_tempimage = np.transpose(input_tempimage,(1,2,0))
        target_tempimage = np.transpose(target_tempimage,(1,2,0))
        plt.imshow(input_tempimage)
        plt.show()
        plt.imshow(target_tempimage)
        plt.show()

        output = Generator(input)
        print(np.array(output.cpu().detach()).shape)
        loss = Vggloss(output,target, show=True)

        break