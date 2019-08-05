from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import GAIL.Generator as Generator
import GAIL.Discriminator as Discriminator


parser = argparse.ArgumentParser()
#Net
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--manualSeed', type=int, help='manual seed')
#Data
parser.add_argument('--dataset', required=False, default='folder')
parser.add_argument('--dataroot', required=False, default='./data', help='path to dataset')
parser.add_argument('--outf', default='./Output', help='folder to output images and model checkpoints')
#GPU
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
#Generator
parser.add_argument('--Gkernel', required=False, default=2, help='AKA filter')
parser.add_argument('--GinChannel', type=int, required=False, default=210)
parser.add_argument('--GoutChannel',type=int, required=False, default=18)
#Discriminator
parser.add_argument('--Dkernel', required=False, default=2, help='AKA filter')
parser.add_argument('--DinChannel', type=int, required=False, default=210)
parser.add_argument('--DoutChannel',type=int, required=False, default=18)
#GAIL
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GAIL():
    def __init__(self,stateSize, actionSize, maxAction)-> None:
        self.policy = Generator(stateSize, actionSize, opt.Gkernel, maxAction).to(device)
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=opt.lr, beta=opt.betas)

        self.discriminator = Discriminator(stateSize, actionSize, opt.Gkernel).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, beta=opt.betas)

        #self.numIntaration = numIteration()

        #self.loss_fn = nn.BCELoss()

    def sample(self):

        print("Loaded experties' trajectories")

    def update(self):
        print("")

    def policyStep(self,state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy(state).cpu().data.numpy().flatten()

    def train(self):
        for x in range(0,self.numIntaration):
            self.sample()
            self.update()
            self.policyStep()


