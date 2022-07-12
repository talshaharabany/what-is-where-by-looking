from base import *
from models.vgg16 import VGG16


class MultiModel(nn.Module):
    def __init__(self, args):
        super(MultiModel, self).__init__()
        self.E = VGG16()
        self.D = MMDecoder(self.E.full_features,
                           out_channel=1,
                           out_size=(int(args['Isize']), int(args['Isize'])),
                           is_blip=False)

    def forward(self, image, z_text):
        z_image = self.E(image)
        mask = self.D(z_image, z_text)
        return mask


