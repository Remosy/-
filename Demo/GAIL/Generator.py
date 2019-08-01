"""
Policy Generator
"""

import torch.nn as nn
import GAIL.SPP as SPP
#https://github.com/NVlabs/SPADE/tree/master/models/networks
#https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py

class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.filter = opt.numFilter
        self.sw, self.sh = self.compute_latent_vector_size(opt)
        self.opt = opt
        self.main = nn.Sequential(
            nn.Linear(self.opt.zDim, 16 * self.filter * self.sw * self.sh),

            nn.Conv2d(self.opt.inChannel, self.opt.outChannel, self.opt.kernel, stride=2, padding=1, bias=False),
            nn.ReLU(True),

            nn.Conv2d(self.opt.outChannel, self.opt.outChannel * 2, self.opt.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.opt.outChannel * 2),
            nn.ReLU(True),

            nn.Conv2d(self.opt.outChannel * 2, self.opt.outChannel * 4, self.opt.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.opt.outChannel * 4),
            nn.ReLU(True),

            nn.Conv2d(self.opt.outChannel * 4, self.opt.outChannel * 8, self.opt.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.opt.outChannel * 8),
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



