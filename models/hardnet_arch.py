import torch
import torch.nn as nn


class HardNet_model(nn.Module):
    def __init__(self, args):
        super(HardNet_model, self).__init__()
        if args['order_ae'] == '68':
            self.Org_model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet68', pretrained=True)
            self.layers = [1, 2, 4, 9, 12, 15]
            self.full_features = [64, 128, 320, 640, 1024]
            self.N = 17
        elif args['order_ae'] == '68ds':
            self.Org_model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet68ds', pretrained=True)
            self.layers = [1, 2, 4, 9, 12, 15]
            self.full_features = [64, 128, 320, 640, 1024]
            self.N = 17
        elif args['order_ae'] == '39':
            self.Org_model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet39ds', pretrained=True)
            self.layers = [1, 2, 4, 7, 10, 13]
            self.full_features = [48, 96, 320, 640, 1024]
            self.N = 15
        elif args['order_ae'] == '85':
            self.Org_model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet85', pretrained=True)
            self.layers = [1, 2, 4, 9, 14, 18]
            self.full_features = [96, 192, 320, 720, 1280]
            self.N = 20
        for param in self.Org_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = []
        for i in range(self.N):
            x = self.Org_model.base[i](x)
            # print((i, x.shape))
            if i in self.layers:
                out.append(x)
        return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-order_ae', '--order_ae', default='39', help='order of the backbone - ae',
                        required=False)
    args = vars(parser.parse_args())
    model = HardNet_model(args=args).cuda()
    x = torch.randn((16, 3, 224, 224)).cuda()
    z = model(x)
    for i in range(6):
        print(z[i].shape)




