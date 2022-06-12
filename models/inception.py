import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class InceptionV3Net(nn.Module):
    def __init__(self):
        super(InceptionV3Net, self).__init__()
        self.Org_model = models.inception_v3(pretrained=True)
        self.full_features = [64, 192, 288, 768, 2048]
        for param in self.Org_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.Org_model.Conv2d_1a_3x3(x)
        x = self.Org_model.Conv2d_2a_3x3(x)
        x = self.Org_model.Conv2d_2b_3x3(x)
        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=True)

        x1 = self.Org_model.maxpool1(x)
        x1 = self.Org_model.Conv2d_3b_1x1(x1)
        x1 = F.interpolate(x1, size=(56, 56), mode="bilinear", align_corners=True)

        x2 = self.Org_model.Conv2d_4a_3x3(x1)
        x1 = x1[:, :64, :, :]
        x2 = F.interpolate(x2, size=(56, 56), mode="bilinear", align_corners=True)

        x3 = self.Org_model.maxpool2(x2)
        x3 = self.Org_model.Mixed_5b(x3)
        x3 = self.Org_model.Mixed_5c(x3)
        x3 = self.Org_model.Mixed_5d(x3)
        x3 = F.interpolate(x3, size=(28, 28), mode="bilinear", align_corners=True)

        x4 = self.Org_model.Mixed_6a(x3)
        x4 = self.Org_model.Mixed_6b(x4)
        x4 = self.Org_model.Mixed_6c(x4)
        x4 = self.Org_model.Mixed_6d(x4)
        x4 = self.Org_model.Mixed_6e(x4)
        x4 = F.interpolate(x4, size=(14, 14), mode="bilinear", align_corners=True)

        x5 = self.Org_model.Mixed_7a(x4)
        x5 = self.Org_model.Mixed_7b(x5)
        x5 = self.Org_model.Mixed_7c(x5)
        x5 = F.interpolate(x5, size=(7, 7), mode="bilinear", align_corners=True)
        return x, x1, x2, x3, x4, x5


if __name__ == "__main__":
    model = InceptionV3Net().cuda()
    x = torch.randn((16, 3, 224, 224)).cuda()
    z = model(x)
    for i in range(6):
        print(z[i].shape)