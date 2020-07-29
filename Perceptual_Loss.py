import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
from torchsummary import summary
import os
from PIL import Image

PATCH_DIR ="Result_image/patchimage"
def savepatch(DIRPATH, batch,layer):

    DIRPATH_LAYER = os.path.join(DIRPATH,"layer{}".format(layer))
    if not os.path.isdir(DIRPATH_LAYER):
        os.mkdir(DIRPATH_LAYER)

    for i,patch in enumerate(batch):
        plt.imshow(patch)
        plt.savefig(os.path.join(DIRPATH_LAYER,"{}th_feature.png".format(i,layer)))

class vggloss(nn.Module):
    def __init__(self,truncate_layer=36):
        super(vggloss, self).__init__()
        self.model = models.vgg.vgg19(pretrained=True)
        self.feature_extraction_model = nn.Sequential(*list(self.model.features)[:truncate_layer]).eval()
        print(list(self.model.features)[:truncate_layer])
        self.truncated_layer = truncate_layer

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
            savepatch(os.path.join(PATCH_DIR,"input"),patch_image_input,self.truncated_layer)
          #  plt.imshow(patch_image_input[:,:,0])
           # plt.show()

            patch_image_target = np.array(target_extracted_fatch.cpu().detach()).squeeze()
            print(patch_image_target.shape)
            savepatch(os.path.join(PATCH_DIR,"target"), patch_image_target, self.truncated_layer)
           # plt.imshow(patch_image_target[:, :, 0])
          #  plt.show()

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