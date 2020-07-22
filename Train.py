import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import Dataset_gen
import Model
import Perceptual_Loss

from tqdm import tqdm

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
    train_dataset = Dataset_gen.Dataset_Train(dirpath=DIRPATH_TRAIN, crop_size=96, upscale_factor=UPSCALIE_FACTOR)
    vaild_dataset = Dataset_gen.Dataset_Vaild(dirpath=DIRPATH_VAILD, upscale_factor=UPSCALIE_FACTOR)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    vaild_dataloader = DataLoader(dataset=vaild_dataset, batch_size=1)

    Gen_Model = Model.Generator()
    Dis_Model = Model.Discriminator()

    Gen_optimizer = optim.Adam(Gen_Model.parameters())
    Dis_optimizer = optim.Adam(Dis_Model.parameters())

    Vggloss = Perceptual_Loss.vggloss()  # vgg(5,4) loss

    Gen_Model = Gen_Model.to(device)
    Dis_Model = Dis_Model.to(device)

    Vggloss = Vggloss.to(device)

    content_criterion = nn.MSELoss()
    adversal_criterion = nn.BCELoss()

    PSNR_eval = np.zeros(TOTAL_EPOCH)
    PSNR_train = np.zeros(TOTAL_EPOCH)
    Train_Gen_loss = np.zeros(TOTAL_EPOCH)
    Train_Dis_loss = np.zeros(TOTAL_EPOCH)
    train_len = len(train_dataloader)

    for epoch in range(TOTAL_EPOCH):
        # prepare training
        Gen_Model.train()
        Dis_Model.train()
        Gen_loss_total = 0
        Dis_loss_total = 0
        total_PSNR_train = 0
        print("----epoch {}/{}----".format(epoch + 1, TOTAL_EPOCH))
        print("----training step----")
        for i, (input, target) in enumerate(train_dataloader):
            print("---batch {}---".format(i))
            target_list = np.array(target)
            input_list = np.array(input)
            for i, target_image in enumerate(target_list):
                print("target {} : {}".format(i,np.array(target_image).shape))
            for i,input_image in enumerate(input):
                print("input {}: {}".format(i,np.array(input_image).shape))

            input, target = input.to(device), target.to(device)


            # train Discriminator
            Dis_optimizer.zero_grad()

            lr_generated = Gen_Model(input)
            lr_discriminated = Dis_Model(lr_generated)
            hr_discriminated = Dis_Model(target)

            Mseloss_temp = content_criterion(lr_generated, target)
            PSNR_temp = 10 * torch.log10(1 / Mseloss_temp)
            total_PSNR_train += PSNR_temp

            adversarial_loss = adversal_criterion(lr_discriminated,
                                                  torch.zeros_like(lr_discriminated)) + adversal_criterion(hr_discriminated,
                                                                                                           torch.ones_like(
                                                                                                               hr_discriminated))

            adversarial_loss.backward()
            Dis_optimizer.step()

            Dis_loss_total += adversarial_loss.item()
            #    if grad_clip is not None:
            #          torch.utils.clip_gradient

            # train Generator
            Gen_optimizer.zero_grad()

            lr_generated = Gen_Model(input)
            lr_discriminated = Dis_Model(lr_generated)

            adversarial_loss = adversal_criterion(lr_discriminated, torch.ones_like(lr_discriminated))
            #  content_loss = content_criterion(Vggloss(target),Vggloss(lr_generated))
            # print(np.array(target.cpu().detach()).shape)
            # print(np.array(lr_generated.cpu().detach()).shape)
            content_loss = Vggloss(target, lr_generated)

            Gen_loss = content_loss + 0.001 * adversarial_loss

            Gen_loss.backward()
            Gen_optimizer.step()
            Gen_loss_total += Gen_loss.item()
            print("training step : {}/{}".format(i + 1, train_len))

        Train_Gen_loss[epoch] = Gen_loss_total / len(train_dataloader)
        Train_Dis_loss[epoch] = Dis_loss_total / len(train_dataloader)
        PSNR_train[epoch] = total_PSNR_train / len(train_dataloader)
        print("train PSNR : {}".format(total_PSNR_train / len(train_dataloader)))

        Gen_Model.eval()
        Dis_Model.eval()
        total_PSNR_eval = 0
        print("----evaluation step----")
        with torch.no_grad():
            # val_bar = tqdm(vaild_dataloader)
            for input, target in vaild_dataloader:
                input = input.to(device)
                fakeimage = Gen_Model(input)
                fakeimage = np.array(fakeimage.cpu().detach())
                fakeimage = fakeimage.squeeze()
                print(fakeimage.shape())

                target = np.array(target.detach())
                batch_MSE = np.mean((fakeimage - target) ** 2)
                PSNR_temp = 10 * torch.log10(1 / batch_MSE)
                total_PSNR_eval += PSNR_temp

            PSNR_eval[epoch] = total_PSNR_eval / len(vaild_dataloader)
            print("evaluation PSNR : {}".format(total_PSNR_eval / len(vaild_dataloader)))

        torch.save(Gen_Model.state_dict(), "Trained_model/Generator/generator_{}th_model.pth".format(epoch))
        torch.save(Dis_Model.state_dict(), "Trained_model/Discriminator/discriminator_{}th_model.pth".format(epoch))

