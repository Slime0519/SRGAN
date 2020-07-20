import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import Dataset_gen
import Model
import Perceptual_Loss

BATCH_SIZE = 16
CROP_SIZE = 96
UPSCALIE_FACTOR = 4
DIRPATH_TRAIN = "Dataset/Train"
DIRPATH_VAILD = "Dataset/Vaild"
TOTAL_EPOCH = 100
grad_clip = None

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu:0")

if __name__ == "__main__":
    train_dataset = Dataset_gen.Dataset_Train(dirpath = DIRPATH_TRAIN, crop_size= 96, upscale_factor= UPSCALIE_FACTOR)
    vaild_dataset = Dataset_gen.Dataset_Vaild(dirpath = DIRPATH_VAILD, upscale_factor= UPSCALIE_FACTOR)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE ,shuffle=True, num_workers=4)
    vaild_dataset = DataLoader(dataset=vaild_dataset, batch_size=1)

    Gen_Model = Model.Generator()
    Dis_Model = Model.Discriminator()

    Gen_optimizer = optim.Adam(Gen_Model.parameters())
    Dis_optimizer = optim.Adam(Dis_Model.parameters())

    Vggloss = Perceptual_Loss.vggloss() #vgg(5,4) loss

    Gen_Model = Gen_Model.to(device)
    Dis_Model = Dis_Model.to(device)

    Vggloss = Vggloss.to(device)

    content_criterion = nn.MSELoss()
    adversal_criterion = nn.BCELoss()

    for epoch in range(TOTAL_EPOCH):
        #prepare training
        Gen_Model.train()
        Dis_Model.train()

        for input, target in train_dataloader:
            input, target  = input.to(device), target.to(device)

            #train Discriminator
            Dis_optimizer.zero_grad()

            lr_generated = Gen_Model(input)
            lr_discriminated = Dis_Model(lr_generated)
            hr_discriminated = Dis_Model(target)

            adversarial_loss = adversal_criterion(lr_discriminated,torch.zeros_like(lr_discriminated)) +  adversal_criterion(hr_discriminated,torch.ones_like(hr_discriminated))

            adversarial_loss.backward()
            Dis_optimizer.step()

        #    if grad_clip is not None:
        #          torch.utils.clip_gradient

            #train Generator
            Gen_optimizer.zero_grad()

            lr_generated = Gen_Model(input)
            lr_generated = Dis_Model(lr_generated)

            adversarial_loss = adversal_criterion(lr_discriminated,torch.ones_like(lr_discriminated))
            content_loss = content_criterion(Vggloss(target),Vggloss(lr_generated))

            Gen_loss = content_loss + 0.001*adversarial_loss

            Gen_loss.backward()
            Gen_optimizer.step()

        Gen_Model.eval()
        Dis_Model.eval()

        with torch.no_grad():
            _












