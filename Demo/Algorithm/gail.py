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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='folder', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=False, default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='./Output', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

policy_parser = argparse.ArgumentParser()
parser.add_argument('--statedim', type=int, help='manual statedim')
parser.add_argument('--actdim', type=int, help='manual actiondim')

discri_parser = argparse.ArgumentParser()
parser.add_argument('--statedim', type=int, help='manual seed')
parser.add_argument('--actdim', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataloader = torch.utils.data.DataLoader(
    dset.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=opt.batchSize, shuffle=True)


nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class policyNet(nn.Module):
    def __init__(self, policy_parser):
        super(policyNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(policy_parser.statedim, 400),
            nn.ReLU(True),

            nn.Linear(400, 300),
            nn.ReLU(True),

            nn.Linear(300, policy_parser.actdim),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class discrinatorNet(nn.Module):
    def __init__(self, discri_parser):
        super(discrinatorNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(discri_parser.statedim + discri_parser.actdim, 400),
            nn.Tanh(),

            nn.Linear(400, 300),
            nn.Tanh(),

            nn.Linear(300, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)


class GAIL():
    def __init__(self,xprtTraj,initPolicy, policy, w,numIteration)-> None:
        self.xprtTraj = xprtTraj

        self.policy = policyNet(policy_parser).to(device)
        self.optim_policy = torch.optim.Adam(self.policy.parameters())

        self.discriminator = discrinatorNet(discri_parser).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters())
        self.numIntaration = numIteration

        self.loss_fn = nn.BCELoss()

    def sample(self):

        print("")

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


