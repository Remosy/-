"""
Policy Generator
"""

import torch.nn as nn
from GAIL.SPP import SPP
from commons.DataInfo import DataInfo
import torch
#https://github.com/NVlabs/SPADE/tree/master/models/networks
#https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Generator(nn.Module):
    """The summary line for a class docstring should fit on one line.

        If the class has public attributes, they may be documented here
        in an ``Attributes`` section and follow the same formatting as a
        function's ``Args`` section. Alternatively, attributes may be documented
        inline with the attribute's declaration (see __init__ method below).

        Properties created with the ``@property`` decorator should be documented
        in the property's getter method.

        Attributes:
            attr1 (str): Description of `attr1`.
            attr2 (:obj:`int`, optional): Description of `attr2`.

        """
    def __init__(self, datainfo:DataInfo):
        super(Generator, self).__init__()
        self.inChannel = datainfo.generatorIn #state space size
        self.outChannel = datainfo.generatorOut #action space size
        self.kernel = datainfo.generatorKernel #number of filter
        self.pyramidLevel = [4, 2, 1] #3-level pyramid
        self.maxAction = datainfo.maxAction
        self.spp = SPP().to(device)
        self.std = 0.0

        self.main = nn.Sequential(
            #Downsampling
            nn.Conv2d(self.inChannel, self.outChannel, kernel_size=self.kernel, stride=2, padding=1, bias=False),
            nn.ReLU(True),

            nn.Conv2d(self.outChannel, self.outChannel * 2, kernel_size=self.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.outChannel * 2),
            nn.ReLU(True),

            nn.Conv2d(self.outChannel * 2, self.outChannel * 4, kernel_size=self.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.outChannel * 4),
            nn.ReLU(True),

            nn.Conv2d(self.outChannel * 4, self.outChannel * 8, kernel_size=self.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.outChannel * 8),
            nn.ReLU(True),
        )
        #self.fc1 = nn.Linear(256, self.outChannel * 8)
        self.fc2 = nn.Linear(self.outChannel * 8, self.outChannel)
        self.softmax = nn.Softmax(dim=1)
        self.log_std = 0

    def forward(self, input):
        midOut = self.main(input)
        sppOut = self.spp.doSPP(midOut, int(midOut.size(0)), [int(midOut.size(2)), int(midOut.size(3))], self.pyramidLevel, self.kernel) # last pooling layer
        del midOut
        fc1 = nn.Linear(sppOut.shape[1], self.outChannel * 8).to(device) #update
        fcOut1 = fc1(sppOut)
        del sppOut
        fcOut2 = self.fc2(fcOut1)
        output = self.softmax(fcOut2)

        self.log_std = nn.Parameter(torch.ones(1, self.outChannel) * self.std)
        self.std = self.log_std.exp().expand_as(output)
        return output



