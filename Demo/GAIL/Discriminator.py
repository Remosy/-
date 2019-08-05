"""
Policy Discriminator
"""

import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, inChannel:int, outChhannel:int, kernel:int):
        super(Discriminator, self).__init__()
        self.inChannel = inChannel  # state space size
        self.outChannel = outChhannel  # action space size
        self.kernel = kernel  # number of filter

        self.main = nn.Sequential(
            nn.Conv2d(self.inChannel, self.outChannel, self.kernel, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(self.outChannel, self.outChannel * 2, self.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.outChannel * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(self.outChannel * 2, self.outChannel * 4, self.kernel, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.outChannel * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(self.outChannel * 4, self.outChannel * 8, self.kernel, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.outChannel * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, input):
        output = self.main(input)
        return output
