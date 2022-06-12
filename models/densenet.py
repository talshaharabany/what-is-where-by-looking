import torch
import torch.nn as nn
from torchvision import models


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.Org_model = models.densenet161(pretrained=True)
        self.full_features = [96, 384, 768, 2112, 2208]
        for param in self.Org_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.Org_model.features.conv0(x)
        x = self.Org_model.features.norm0(x)
        x = self.Org_model.features.relu0(x)
        x1 = self.Org_model.features.pool0(x)
        x2 = self.Org_model.features.denseblock1(x1)
        x3 = self.Org_model.features.transition1(x2)
        x3 = self.Org_model.features.denseblock2(x3)
        x4 = self.Org_model.features.transition2(x3)
        x4 = self.Org_model.features.denseblock3(x4)
        x5 = self.Org_model.features.transition3(x4)
        x5 = self.Org_model.features.denseblock4(x5)
        return x, x1, x2, x3, x4, x5


if __name__ == "__main__":
    model = DenseNet().cuda()
    x = torch.randn((16, 3, 224, 224)).cuda()
    z = model(x)
    for i in range(6):
        print(z[i].shape)