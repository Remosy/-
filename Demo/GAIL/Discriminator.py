"""
Policy Discriminator
"""

import torch.nn as nn
from commons.DataInfo import DataInfo

class Discriminator(nn.Module):

    def __init__(self, datainfo:DataInfo):
        super(Discriminator, self).__init__()
        self.inChannel = datainfo.discriminatorIn  # action
        self.outChannel = datainfo.discriminatorOut  # binary
        self.kernel = datainfo.discriminatorKernel  # number of filter

        self.main = nn.Sequential(
            nn.Conv1d(self.inChannel, self.outChannel, self.kernel, stride=2, padding=1, bias=False), #require 3D input
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv1d(self.outChannel, self.outChannel * 2, self.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.outChannel * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv1d(self.outChannel * 2, self.outChannel * 4, self.kernel, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm1d(self.outChannel * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv1d(self.outChannel * 4, self.outChannel * 8, self.kernel, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm1d(self.outChannel * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, input):
        output = self.main(input) #ToDo: got 2D want 3D
        return output
