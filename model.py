from base import *
from resnet import *
from models.vgg16 import VGG16Net
from models.mobilenet import MobileNetV1
from models.inception import *
from models.densenet import *
from models.hardnet_arch import *
import numpy as np

def unify(za, zb, alpha):
    z = []
    for i in range(len(za)):
        tmp = alpha * za[i] + (1 - alpha) * zb[i]
        z.append(tmp)
    return z


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args['backbone'] == 'resnet':
            self.E = ResNet(args=args)
        elif args['backbone'] == 'vgg':
            self.E = VGG16Net()
        elif args['backbone'] == 'mobile':
            self.E = MobileNetV1()
        elif args['backbone'] == 'inception':
            self.E = InceptionV3Net()
        elif args['backbone'] == 'densenet':
            self.E = DenseNet()
        elif args['backbone'] == 'hardnet':
            self.E = HardNet_model(args=args)
        self.D = SkipDecoder(self.E.full_features,
                             out_channel=1,
                             out_size=(int(args['Isize']), int(args['Isize'])))

    def forward(self, a):
        # alpha = 0.2 * torch.rand((a.shape[0], 1, 1, 1)).cuda() + 0.8
        za = self.E(a)
        # zb = self.E(
        # z = unify(za,zb, alpha)
        # M = F.sigmoid(self.D(z))
        Ma = F.sigmoid(self.D(za))
        return Ma


class MultiModel(nn.Module):
    def __init__(self, args):
        super(MultiModel, self).__init__()
        if args['backbone'] == 'resnet':
            self.E = ResNet(args=args)
        elif args['backbone'] == 'vgg':
            self.E = VGG16Net()
        elif args['backbone'] == 'mobile':
            self.E = MobileNetV1()
        elif args['backbone'] == 'inception':
            self.E = InceptionV3Net()
        elif args['backbone'] == 'densenet':
            self.E = DenseNet()
        elif args['backbone'] == 'hardnet':
            self.E = HardNet_model(args=args)
        self.D = MMDecoder(self.E.full_features,
                           out_channel=1,
                           out_size=(int(args['Isize']), int(args['Isize'])),
                           is_blip=bool(int(args['is_blip']))
                           )

    def forward(self, image, z_text):
        z_image = self.E(image)
        mask = self.D(z_image, z_text)
        return mask


class MixUpModel(nn.Module):
    def __init__(self, args):
        super(MixUpModel, self).__init__()
        self.E = VGG16Net()
        self.D = MMDecoder(self.E.full_features,
                           out_channel=1,
                           out_size=(int(args['Isize']), int(args['Isize'])),
                           is_blip=False)

    def forward(self, i_a, i_b, z_ta, z_tb, alpha=None):
        if alpha is None:
            shape = (i_a.shape[0], 1, i_a.shape[2], i_a.shape[3])
            alpha = torch.randint(0, 1000, shape).cuda().float() / 1000
        image = alpha * i_a + (1-alpha) * i_b
        z_image = self.E(image)
        m_a = self.D(z_image, z_ta)
        m_b = self.D(z_image, z_tb)
        return m_a, m_b, alpha


if __name__ == "__main__":
    import argparse
    import CLIP.clip as clip
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-backbone', '--backbone', default='vgg', help='order of the backbone - ae',
                        required=False)
    parser.add_argument('-order_ae', '--order_ae', default='16', help='order of the backbone - ae',
                        required=False)
    parser.add_argument('-Isize', '--Isize', default=512, help='image size', required=False)
    args = vars(parser.parse_args())

    def norm_z(z):
        return z / z.norm(dim=1)

    gpu_num = torch.cuda.device_count()
    model = MultiModel(args=args).cuda()
    model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    text = ['image of a man']
    text = clip.tokenize(text).to(device)
    z_t = norm_z(clip_model.encode_text(text))


    x = torch.randn((3, 3, 512, 512)).cuda()
    z = model(x, z_t)