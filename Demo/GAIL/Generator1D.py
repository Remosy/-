#//////////////#####///////////////
#
# ANU u6325688 Yangyang Xu
# Supervisor: Dr.Penny Kyburz
# SPP used in this scrip is adopted some methods from :
# https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py
#//////////////#####///////////////

"""
Policy Generator
"""

import torch.nn as nn
from GAIL.SPP import SPP
from commons.DataInfo import DataInfo
from torch.distributions import Categorical
import torch.nn.functional as F

import torch
from torch.distributions import Normal, Beta


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Generator(nn.Module):

    def __init__(self, datainfo:DataInfo):
        super(Generator, self).__init__()
        self.inChannel = datainfo.generatorIn #state space size
        self.outChannel = datainfo.generatorOut #action space size
        self.maxAction = datainfo.maxAction
        self.criticScore = 0

        self.hidden = torch.nn.Linear(self.inChannel, self.inChannel*2)
        self.out = torch.nn.Linear(self.inChannel*2, self.outChannel)

    def forward(self, input):
        mid = self.hidden(input)
        hOut = F.sigmoid(mid)
        out = self.out(hOut)
        # Critic's
        criticFC = nn.Linear(self.outChannel, 1).to(device)
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





