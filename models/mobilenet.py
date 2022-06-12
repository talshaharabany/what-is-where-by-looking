import torch
import torch.nn as nn


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)
        self.layers = [1, 2, 3, 5, 11, 13]
        self.full_features = [128, 128, 256, 512, 1024]

    def forward(self, x):
        out = []
        for i in range(14):
            x = self.model[i](x)
            if i in self.layers:
                if i == 1:
                    out.append(torch.cat((x, x), dim=1))
                    continue
                out.append(x)
        return out


if __name__ == "__main__":
    model = torch.nn.DataParallel(MobileNetV1()).cuda()
    pretrained = torch.load(r'results/init/cub/mobileV1/mobilenet_sgd_68.848.pth.tar')
    model.load_state_dict(pretrained['state_dict'])
    x = torch.randn((16, 3, 224, 224)).cuda()
    z = model(x)
    state_dict = model.module.state_dict()
    torch.save(state_dict, r'results/init/cub/mobileV1/net_init.pth')
    print(state_dict.keys())