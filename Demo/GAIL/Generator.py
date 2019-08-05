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
    def __init__(self,inChannel,outChhannel,kernel):
        super(Generator, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChhannel
        self.kernel = kernel


        #self.sw, self.sh = self.compute_latent_vector_size(opt)
        self.main = nn.Sequential(
            #nn.Linear(self.opt.zDim, 16 * self.filter * self.sw * self.sh),

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
        self.fc1 = nn.Linear(10752, 4096)
        self.fc2 = nn.Linear(4096, 1000)


    def compute_latent_vector_size(self, opt):
        sw = opt.crop_size // (2 ** opt.numUplayer)
        sh = round(sw / opt.aspRatio)
        return sw, sh

    def forward(self, input):
        midOut = self.main(input)
        sppOut = SPP(midOut,1,[int(midOut.size(2)),int(midOut.size(3))],self.opt.numOutput)
        fcOut1 = self.fc1(sppOut)
        fcOut2 = self.fc2(fcOut1)
        output = nn.Tanh(fcOut2)
        return output



