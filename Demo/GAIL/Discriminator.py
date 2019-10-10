"""
Policy Discriminator
"""

import torch.nn as nn
from commons.DataInfo import DataInfo
import numpy as np
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, datainfo:DataInfo):
        super(Discriminator, self).__init__()
        self.inChannel = datainfo.discriminatorIn  # action
        self.outChannel = datainfo.discriminatorOut  # binary
        self.kernel = datainfo.discriminatorKernel  # number of filter
        self.imgSize = np.prod(datainfo.stateShape)
        #self.Lsize = self.imgSize//2**4
        self.main = nn.Sequential(
            F.tanh(nn.Linear(self.inChannel,self.outChannel)),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output
