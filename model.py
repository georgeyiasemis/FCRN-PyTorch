import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from blocks import *


class FCRN(nn.Module):
    '''
    FCRN: Fully Convolutional Residual Network based on a ResNet-50 architecture.

    Implementation follows architecture of FCRN in "Deeper Depth Prediction with
    Fully Convolutional Residual Networks" by Laina I. et al.
    '''
    def __init__(self, in_channels):
        super(FCRN, self).__init__()

        modules = OrderedDict()
        encoder = OrderedDict()
        decoder = OrderedDict()
        # Encoder
        # In Convolution Layer
        conv1 = nn.Conv2d(in_channels=in_channels,
                          out_channels=64,
                          kernel_size=(7, 7),
                          stride=(2, 2),
                          padding=(3, 3))
        batchnorm1 = nn.BatchNorm2d(64)
        maxpool1 = nn.MaxPool2d(kernel_size=(3, 3),
                                stride=(2, 2),
                                padding=(1, 1))
        relu = nn.ReLU()
        encoder['Conv1'] = nn.Sequential(conv1, batchnorm1, maxpool1, relu)

        # Skip Connection Layers
        encoder['Skip1'] = SkipConvBlock(num_blocks=3, in_channels=64, d1=64)
        encoder['Skip2'] = SkipConvBlock(num_blocks=4, in_channels=64 * 4, d1=128, stride=2)
        encoder['Skip3'] = SkipConvBlock(num_blocks=6, in_channels=128 * 4, d1=256, stride=2)
        encoder['Skip4'] = SkipConvBlock(num_blocks=3, in_channels=256 * 4, d1=512, stride=2)

        # Convolution Layer
        conv2 = nn.Conv2d(in_channels=512 * 4,
                          out_channels=512 * 2,
                          kernel_size=(1, 1))
        batchnorm2 = nn.BatchNorm2d(512 * 2)

        encoder['Conv2'] = nn.Sequential(conv2, batchnorm2)

        # Decoder
        # Fast Up Projection Blocks
        decoder['UpProj1'] = FastUpProjectionBlock(in_channels=512 * 2, out_channels=512)
        decoder['UpProj2'] = FastUpProjectionBlock(in_channels=256 * 2, out_channels=256)
        decoder['UpProj3'] = FastUpProjectionBlock(in_channels=128 * 2, out_channels=128)
        decoder['UpProj4'] = FastUpProjectionBlock(in_channels=64 * 2, out_channels=64)

        # Out Convolution Layer
        conv3 = nn.Conv2d(in_channels=64,
                          out_channels=1,
                          kernel_size=(3, 3),
                          padding=(1, 1))
        decoder['Conv3'] = nn.Sequential(conv3, relu)
        modules['Encoder'] = nn.Sequential(encoder)
        modules['Decoder'] = nn.Sequential(decoder)
        self.net = nn.Sequential(modules)


    def forward(self, inp):
        '''
        inp: Tensor of shape (batch_size, in_channels, H, W)
        out: Tensor of shape (batch_size, 1, H', W')
        '''

        return self.net(inp)

if __name__ == '__main__':
    in_channels = 3
    x = torch.randn(10, in_channels, 304, 228)
    fcrn = FCRN(in_channels=in_channels)
    x = fcrn(x)
    print(x.shape)
