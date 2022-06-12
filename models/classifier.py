import torch
import torch.nn as nn
from torchvision import models


class classifier(nn.Module):
    def __init__(self, args):
        super(classifier, self).__init__()
        res = int(args['order'])
        if res == 18:
            emb_size = 512
        elif res == 34:
            emb_size = 512
        elif res == 50:
            emb_size = 2048
        elif res == 101:
            emb_size = 2048
        self.E = ResNet(res=res, args=args)
        self.fc1 = nn.Linear(emb_size, int(args['nC']))

    def forward(self, x, flag=False):
        if flag:
            x1, _ = self.E(x)
            x = self.fc1(x1)
            return x, x1
        else:
            x, _ = self.E(x)
            x = self.fc1(x)
            return x


class ResNet(nn.Module):
    def __init__(self, res=50, is_grad=True, args=None):
        super(ResNet, self).__init__()
        if res == 18:
            self.Org_model = models.resnet18(pretrained=True)
        if res == 34:
            self.Org_model = models.resnet34(pretrained=True)
        if res == 50:
            self.Org_model = models.resnet50(pretrained=True)
        if res == 101:
            self.Org_model = models.resnet101(pretrained=True)
        for param in self.Org_model.parameters():
            param.requires_grad = is_grad

    def forward(self, x_in):
        x = self.Org_model.conv1(x_in)
        x = self.Org_model.bn1(x)
        x = self.Org_model.relu(x)
        x = self.Org_model.maxpool(x)
        x1 = self.Org_model.layer1(x)
        x2 = self.Org_model.layer2(x1)
        x3 = self.Org_model.layer3(x2)
        x4 = self.Org_model.layer4(x3)
        x5 = self.Org_model.avgpool(x4).squeeze()
        return x5, x4


