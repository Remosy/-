"""
Policy Discriminator
"""

import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.filter = opt.numFilter
        self.sw, self.sh = self.compute_latent_vector_size(opt)
        self.opt = opt
        self.main = nn.Sequential(
            nn.Conv2d(self.opt.inChannel, self.opt.outChannel, self.opt.kernel, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(self.opt.outChannel, self.opt.outChannel * 2, self.opt.kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.opt.outChannel * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(self.opt.outChannel * 2, self.opt.outChannel * 4, self.opt.kernel, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.opt.outChannel * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(self.opt.outChannel * 4, self.opt.outChannel * 8, self.opt.kernel, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.opt.outChannel * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, input):
        output = self.main(input)
        return output
