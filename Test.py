import torch
import torch.nn as nn
import torch.utils as utils
import Model
import Dataset_gen
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    testset_dirpath = "Dataset/Test"
    testset_name = "BSDS300"

    model_dirpath = "Trained_model"
    model_epoch = 100

    savedir = "Result_image"

    gen_model = Model.Generator()
    Test_Dataset = Dataset_gen.Dataset_Test(dirpath=os.path.join(testset_dirpath,testset_name))
    Test_Dataloader = DataLoader(dataset=Test_Dataset, shuffle=False, batch_size=1, num_workers=0)

    gen_model.load_state_dict(torch.load(os.path.join(model_dirpath,"Generator","generator_{}th_model.pth".format(model_epoch-1))))
    gen_model = gen_model.to(device)
    gen_model.eval()

    for i,input in enumerate(Test_Dataloader):
        input = input.to(device)
        output = gen_model(input)
        output_image = np.array(output.cpu().detach())
        output_image = output_image.squeeze()
        output_image = np.transpose(output_image,(1,2,0))
        output_image = (output_image*255).astype(np.uint8)
        if i % 10 == 0:
            print(np.array(input.cpu().detach()).shape)
            print(output_image.shape)
            plt.imshow(output_image)
            plt.show()
            plt.imshow(output_image)
            plt.savefig(os.path.join(savedir,"resultimage_{}_{}".format(testset_name,i)),dpi = 500)

            input_temp = np.array(input.cpu().detach())
            input_bicubic = cv2.resize(np.transpose(np.squeeze(input_temp),(1,2,0)),dsize=(0,0),fx = 4, fy = 4, interpolation=cv2.INTER_CUBIC)
            plt.imshow(input_bicubic)
            plt.show()

    PSNR_eval = np.load("result_data/PSNR_eval.npy")
    PSNR_Train = np.load("result_data/PSNR_train.npy")
    Train_Dis_loss = np.load("result_data/Train_Dis_loss.npy")
    Train_Gen_loss = np.load("result_data/Train_Gen_loss.npy")

    Num_Epoch = len(PSNR_eval)
    x= range(Num_Epoch)
    y_eval = PSNR_eval
    y_train = PSNR_Train
    plt.plot(x,y_train)
    plt.plot(x,y_eval)
    plt.legend(['train PSNR', 'evaluation PSNR'])
    plt.title("average PSNR at train and evaluation step")
    plt.show()

    dis_loss = Train_Dis_loss
    gen_loss = Train_Gen_loss

    plt.plot(x,dis_loss)
    plt.plot(x,gen_loss)
    plt.legend(['discriminator loss', 'generator loss'])
    plt.title("average loss of generator and discriminator loss at training step")
    plt.show()



