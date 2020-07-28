import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
from torchsummary import summary

PATCH_DIR ="Result_image/patchimage"

class vggloss(nn.Module):
    def __init__(self,truncate_layer=36):
        super(vggloss, self).__init__()
        self.model = models.vgg.vgg19(pretrained=True)
        self.feature_extraction_model = nn.Sequential(*list(self.model.features)[:truncate_layer+1]).eval()

        for param in self.feature_extraction_model.parameters():
            param.requires_grad = False
        self.MSE_Loss = nn.MSELoss()

    def forward(self,input,target,show = False):
        input_extracted_fatch = self.feature_extraction_model(input)
        target_extracted_fatch = self.feature_extraction_model(target)
        perception_loss = self.MSE_Loss(input_extracted_fatch,target_extracted_fatch)

        if show:
            patch_image_input = np.array(input_extracted_fatch.cpu().detach()).squeeze()
            print(patch_image_input.shape)
            patch_image_input = np.transpose(patch_image_input,(1,2,0))
            plt.imshow(patch_image_input[:,:,0])
            plt.show()

            patch_image_target = np.array(target_extracted_fatch.cpu().detach()).squeeze()
            print(patch_image_target.shape)
            patch_image_target = np.transpose(patch_image_target, (1, 2, 0))
            plt.imshow(patch_image_target[:, :, 0])
            plt.show()

        return perception_loss
    """
    # test용 forward 함수
    def forward(self,x):
        out = self.model(x)
        return out
    """
if __name__ == "__main__":

    Test = vggloss().to('cuda:0')
    #summary(Test,(3,384,384))