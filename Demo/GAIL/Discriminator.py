#//////////////#####///////////////
#
# ANU u6325688 Yangyang Xu
# Supervisor: Dr.Penny Kyburz
#//////////////#####///////////////
"""
Policy Discriminator
"""
import torch.nn as nn
from commons.DataInfo import DataInfo
import numpy as np

###################################################################################
#
#  DISCRIMINATOR for 3D LOCATION
#
###################################################################################
class Discriminator(nn.Module):

    def __init__(self, datainfo:DataInfo):
        super(Discriminator, self).__init__()
        self.inChannel = datainfo.discriminatorIn  # action
        self.outChannel = datainfo.discriminatorOut  # binary
        self.kernel = datainfo.discriminatorKernel  # number of filter
        self.imgSize = np.prod(datainfo.stateShape)

        self.main = nn.Sequential(
            nn.Linear(self.inChannel, self.outChannel * 16),
            nn.LeakyReLU(0.2, True),

            nn.Linear(self.outChannel * 16, self.outChannel * 2),
            nn.LeakyReLU(0.2, True),

            nn.Linear(self.outChannel * 2, self.outChannel),
            nn.LeakyReLU(0.2, True),

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
        super(Discriminator1D, self).__init__()
        self.inChannel = datainfo.discriminatorIn  # action
        self.outChannel = datainfo.discriminatorOut  # binary
        self.kernel = datainfo.discriminatorKernel  # number of filter
        self.imgSize = np.prod(datainfo.stateShape)

        self.main = nn.Sequential(
            nn.Linear(self.inChannel,self.outChannel*16),
            nn.LeakyReLU(0.2, True),

            nn.Linear(self.outChannel*16, self.outChannel*2),
            nn.LeakyReLU(0.2, True),

            nn.Linear(self.outChannel * 2, self.outChannel),
            nn.LeakyReLU(0.2, True),

            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output
