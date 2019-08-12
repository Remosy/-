from __future__ import print_function
import argparse
import os
import queue
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import gym_recording.playback
import numpy as np
import GAIL.Generator as Generator
import GAIL.Discriminator as Discriminator
import Stage1.getVideoWAction as GetVideoWAction

parser = argparse.ArgumentParser()
#Net
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--betas', type=float, default=0.5, help='beta1 for adam. default=0.5')
#Data
parser.add_argument('--dataset', required=False, default='folder')
parser.add_argument('--dataroot', required=False, default='./data', help='path to dataset')
parser.add_argument('--outf', default='./Output', help='folder to output images and model checkpoints')
#GPU
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
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
        self.expertState = []
        self.expertAction = []
        self.expertReward = []
        self.generator = Generator(stateSize, actionSize, opt.Gkernel, maxAction).to(device)
        self.generatorOptim = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, beta=opt.betas)

        self.discriminator = Discriminator(stateSize, actionSize, opt.Gkernel).to(device)
        self.discriminatorOptim = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, beta=opt.betas)

        #self.numIntaration = numIteration()

        #self.loss_fn = nn.BCELoss()

    def sample(self, folder, targetFolder):
        #load images
        expertData = GetVideoWAction("IceHockey-v0", 3, True)
        exp_state, exp_actions, exp_reward = expertData.replay(folder, targetFolder)
        self.expertState = exp_state
        self.expertAction = exp_actions
        self.expertReward = exp_reward



    def update(self, n_iter, batch_size = 100):
        for i in range(n_iter):
            #######################
            # update discriminator
            #######################
            self.discriminatorOptim.zero_grad()

            # label tensors
            exp_label = torch.full((batch_size, 1), 1, device=device)
            policy_label = torch.full((batch_size, 1), 0, device=device)

            # with expert transitions
            prob_exp = self.discriminator(self.expertState, self.expertAction)
            loss = self.loss_fn(prob_exp, exp_label)

            # with policy transitions
            prob_policy = self.discriminator(state, action.detach())
            loss += self.loss_fn(prob_policy, policy_label)

            # take gradient step
            loss.backward()
            self.optim_discriminator.step()

            ################
            # update policy
            ################
            self.optim_actor.zero_grad()

            loss_actor = -self.discriminator(state, action)
            loss_actor.mean().backward()
            self.optim_actor.step()
        print("")

    def policyStep(self,state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy(state).cpu().data.numpy().flatten()

    def train(self):
        self.sample()
        self.numExprtData = len(self.expertState)
        for x in range(0):
            self.update()
            self.policyStep()

if __name__ == "__main__":
    gail = GAIL()
    gail.sample("../Stage1/openai.gym.1563812853.178726.40887","../reseources")
