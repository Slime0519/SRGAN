import torch
import torch.nn as nn
import torch.utils as utils
import Model
import Dataset_gen
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    testset_dirpath = ""
    testset_name = ""

    model_dirpath = ""
    model_name = ""

    gen_model = Model.Generator()
    Test_Dataset = Dataset_gen.Dataset_Test(dirpath=os.path.join(testset_dirpath,testset_name))
    Test_Dataloader = DataLoader(dataset=Test_Dataset, shuffle=False, batch_size=1, num_workers=0)

    gen_model.load_state_dict(torch.load(os.path.join(model_dirpath,model_name)))
    gen_model = gen_model.to(device)
    gen_model.eval()

    for i,input in enumerate(Test_Dataloader):
        output = gen_model(input)
        output_image = np.array(output.cpu().detach())
        output_image = output_image.squeeze()
        output_image = np.transpose(output_image,(1,2,0))

        if i % 10 == 0:
            plt.imshow(output_image)
            plt.show()


