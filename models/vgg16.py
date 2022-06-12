import torch
import torch.nn as nn
from torchvision import models


class VGG16Net(nn.Module):
    def __init__(self):
        super(VGG16Net, self).__init__()
        self.Org_model = models.vgg16(pretrained=True)
        self.full_features = [64, 128, 256, 512, 512]
        for param in self.Org_model.parameters():
            param.requires_grad = True
        self.layers = [3, 4, 8, 15, 22, 29]
        for param in self.Org_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = []
        for i in range(30):
            x = self.Org_model.features[i](x)
            if i in self.layers:
                out.append(x)
        return out


if __name__ == "__main__":
    model = VGG16Net().cuda()
    x = torch.randn((16, 3, 224, 224)).cuda()
    z = model(x)
