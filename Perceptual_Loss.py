import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

from torchsummary import summary


class vggloss(nn.Module):
    def __init__(self,truncate_layer=36):
        super(vggloss, self).__init__()
        self.model = models.vgg.vgg19(pretrained=True)
        self.feature_extraction_model = nn.Sequential(*list(self.model.features)[:truncate_layer+1]).eval()

        for param in self.feature_extraction_model.parameters():
            param.requires_grad = False
        self.MSE_Loss = nn.MSELoss()

    def forward(self,input,target):
        perception_loss = self.MSE_Loss(self.feature_extraction_model(input),self.feature_extraction_model(target))
        return perception_loss

    """ test용 forward 함수
    def forward(self,x):
        out = self.model(x)
        return out
    """
if __name__ == "__main__":
    Test = vggloss().to('cuda:0')
    # summary(Test,(3,384,384))