"""
Policy Generator
"""

import torch.nn as nn
import GAIL.SPP as SPP
#https://github.com/NVlabs/SPADE/tree/master/models/networks
#https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py


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
    def __init__(self, inChannel:int, outChhannel:int, kernel:int, maxAction:int):
        super(Generator, self).__init__()
        self.inChannel = inChannel #state space size
        self.outChannel = outChhannel #action space size
        self.kernel = kernel #number of filter
        self.pyramidLevel = [4, 2, 1] #3-level pyramid
        self.maxAction = maxAction

        self.main = nn.Sequential(
            #Downsampling
            nn.Conv2d(self.inChannel, self.outChannel, kernel_size=self.kernel, stride=2, padding=1, bias=False),
            nn.ReLU(True),

            nn.Conv2d(self.outChannel, self.outChannel * 2, kernel_size=self.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.outChannel * 2),
            nn.ReLU(True),

            nn.Conv2d(self.outChannel * 2, self.outChannel * 4, kernel_size=self.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.opt.outChannel * 4),
            nn.ReLU(True),

            nn.Conv2d(self.outChannel * 4, self.outChannel * 8, kernel_size=self.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.outChannel * 8),
            nn.ReLU(True),
        )
        self.fc1 = nn.Linear(self.outChannel*8, self.outChannel*4)
        self.fc2 = nn.Linear(self.outChannel*4, self.outChannel)

    def forward(self, input):
        midOut = self.main(input)
        sppOut = SPP(midOut, 1, [int(midOut.size(2)), int(midOut.size(3))], self.pyramidLevel)
        fcOut1 = self.fc1(sppOut)
        fcOut2 = self.fc2(fcOut1)
        output = nn.Tanh(fcOut2)*self.maxAction
        return output



