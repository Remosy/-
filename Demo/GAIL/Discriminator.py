"""
Policy Discriminator
"""

import torch.nn as nn
from commons.DataInfo import DataInfo
import numpy as np

###################################################################################
#
#  DISCRIMINATOR for 1D LOCATION
#
###################################################################################
class Discriminator(nn.Module):

    def __init__(self, datainfo:DataInfo):
        super(Discriminator, self).__init__()
        self.inChannel = datainfo.discriminatorIn  # action
        self.outChannel = datainfo.discriminatorOut  # binary
        self.kernel = datainfo.discriminatorKernel  # number of filter
        self.imgSize = np.prod(datainfo.stateShape)
        #self.Lsize = self.imgSize//2**4
        self.main = nn.Sequential(
            nn.Linear(self.inChannel,self.outChannel),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output


###################################################################################
#
#  DISCRIMINATOR for 1D LOCATION
#
###################################################################################
class Discriminator1D(nn.Module):

    def __init__(self, datainfo:DataInfo):
        super(Discriminator, self).__init__()
        self.inChannel = datainfo.discriminatorIn  # action
        self.outChannel = datainfo.discriminatorOut  # binary
        self.kernel = datainfo.discriminatorKernel  # number of filter
        self.imgSize = np.prod(datainfo.stateShape)
        #self.Lsize = self.imgSize//2**4
        self.main = nn.Sequential(
            nn.Linear(self.inChannel,self.outChannel),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output
