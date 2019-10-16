"""
Policy Generator
"""

import torch.nn as nn
from GAIL.SPP import SPP
from commons.DataInfo import DataInfo
from torch.distributions import Categorical

import torch
from torch.distributions import Normal, Beta
#https://github.com/NVlabs/SPADE/tree/master/models/networks
#https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###################################################################################
#
#  GENERATOR for 3D RGB IMAGE
#
###################################################################################
class Generator(nn.Module):
    def __init__(self, datainfo:DataInfo):
        super(Generator, self).__init__()
        self.inChannel = datainfo.generatorIn #state space size
        self.outChannel = datainfo.generatorOut #action space size
        self.kernel = datainfo.generatorKernel #number of filter
        self.pyramidLevel = [4, 2, 1] #3-level pyramid
        self.maxAction = datainfo.maxAction
        self.spp = SPP().to(device)
        self.mean = 0.0
        self.std = 0.0
        self.criticScore = 0

        self.main = nn.Sequential(
            #Downsampling
            nn.Conv2d(self.inChannel, self.outChannel * 8, kernel_size=self.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.outChannel * 8),
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(self.outChannel * 8, self.outChannel * 4, kernel_size=self.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.outChannel * 4),
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(self.outChannel * 4, self.outChannel * 2, kernel_size=self.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.outChannel * 2),
            nn.LeakyReLU(0.2,True),
        )

        self.fc2 = nn.Linear(self.outChannel * 2, self.outChannel)
        self.softmax = nn.Softmax(dim=1)
        self.log_std = 0

    def forward(self, input):
        midOut = self.main(input)
        sppOut = self.spp.doSPP(midOut, int(midOut.size(0)), [int(midOut.size(2)), int(midOut.size(3))], self.pyramidLevel, self.kernel) # last pooling layer
        del midOut
        fc1 = nn.Linear(sppOut.shape[1], self.outChannel * 2).to(device) #update
        fcOut1 = fc1(sppOut)
        del sppOut
        fcOut2 = self.fc2(fcOut1)
        # Critic's
        criticFC = nn.Linear(self.outChannel, 1).to(device)
        self.criticScore = criticFC(fcOut2)

        # Generator's
        actionDistribution = self.softmax(fcOut2)
        action = (actionDistribution).argmax(1)


        for x in range(actionDistribution.shape[0]):
            if sum(actionDistribution[x]) == 0:
                actionDistribution[x] = actionDistribution[x] + 1e-8

        tmp = Categorical(actionDistribution)
        actionDistribution = tmp.log_prob(action)
        entropy = tmp.entropy()
        return actionDistribution, action.detach(), entropy, fcOut2

###################################################################################
#
#  GENERATOR for 1D LOCATION
#
###################################################################################
class Generator1D(nn.Module):

    def __init__(self, datainfo:DataInfo):
        super(Generator1D, self).__init__()
        self.inChannel = datainfo.generatorIn #state space size
        self.outChannel = datainfo.generatorOut #action space size
        self.criticScore = 0
        self.hidden = nn.Sequential(
            nn.Linear(self.inChannel, self.outChannel * 16),
            nn.LeakyReLU(0.2, True),

            nn.Linear(self.outChannel*16, self.outChannel * 2),
            nn.LeakyReLU(0.2, True)
        )

        self.out = torch.nn.Linear(self.outChannel * 2, self.outChannel)

        self.softmax = nn.Softmax(dim=1)
        self.log_std = 0

    def forward(self, input):
        mid = self.hidden(input)
        out = self.out(mid)
        # Critic's
        criticFC = nn.Linear(self.outChannel * 2, 1).to(device)
        self.criticScore = criticFC(mid)
        # Generator's
        actionDistribution = self.softmax(out)
        action = (actionDistribution).argmax(1)

        for x in range(actionDistribution.shape[0]):
            if sum(actionDistribution[x]) == 0:
                actionDistribution[x]= actionDistribution[x] + 1e-8

        tmp = Categorical(actionDistribution)
        actionDistribution = tmp.log_prob(action)
        entropy = tmp.entropy()
        return actionDistribution, action.detach(), entropy
