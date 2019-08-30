from __future__ import print_function
import argparse
import os
import queue
import random
import torch
import shutil
import glob
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import gym_recording.playback
import numpy as np
from GAIL.Discriminator import Discriminator
from GAIL.Generator import Generator
from Stage1.getVideoWAction import GetVideoWAction
import cv2

#parser = argparse.ArgumentParser()
#Net
#parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
#parser.add_argument('--betas', type=float, default=0.5, help='beta1 for adam. default=0.5')
lr = 0.0002
betas = 0.5
#Data
#parser.add_argument('--dataset', required=False, default='folder')
#parser.add_argument('--dataroot', required=False, default='./data', help='path to dataset')
#parser.add_argument('--outf', default='./Output', help='folder to output images and model checkpoints')
#GPU
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
#parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
#Generator
#parser.add_argument('--Gkernel', required=False, default=2, help='AKA filter')
#parser.add_argument('--GinChannel', type=int, required=False, default=210)
#parser.add_argument('--GoutChannel',type=int, required=False, default=18)
Gkernel = 2
#Discriminator
#parser.add_argument('--Dkernel', required=False, default=2, help='AKA filter')
#parser.add_argument('--DinChannel', type=int, required=False, default=210)
#parser.add_argument('--DoutChannel',type=int, required=False, default=18)
Dkernel = 2
#GAIL
#parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
#parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
#parser.add_argument('--ngf', type=int, default=64)
#parser.add_argument('--ndf', type=int, default=64)
#parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')

#opt = parser.parse_args()
#print(opt)

#try:
    #os.makedirs(opt.outf)
#except OSError:
    #pass

cudnn.benchmark = True

#if torch.cuda.is_available() and not opt.cuda:
    #print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#nz = int(opt.nz)
#ngf = int(opt.ngf)
#ndf = int(opt.ndf)
#nc = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GAIL():
    def __init__(self,folder,target)-> None:
        self.expertState = []
        self.expertAction = []
        self.expertReward = []
        self.sample(folder,target)

        f = cv2.imread(self.expertState[0])

        self.stateShape = f.shape[2]
        self.actionShape = 2
        self.disInShape = f.shape[0]*f.shape[1]*f.shape[2]*2+self.actionShape

        self.maxAction = max(self.expertAction)

        self.generator = Generator(self.stateShape, self.actionShape,Gkernel, self.maxAction).to(device)
        self.generatorOptim = torch.optim.Adam(self.generator.parameters(),lr=lr)

        self.discriminator = Discriminator(self.disInShape, self.actionShape, Dkernel).to(device)
        self.discriminatorOptim = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        #self.numIntaration = numIteration()

        #self.loss_fn = nn.BCELoss()

    def sample(self, folder, targetFolder):
        #load images
        #expertData = GetVideoWAction("IceHockey-v0", 3, True)
        #dataName = expertData.replay(folder, targetFolder)

        dataName ="resources/openai.gym.1566264389.031848.82365"
        #Read Action
        self.expertAction = np.load(dataName+"/action.npy")
        # Read Reward
        self.expertReward = np.load(dataName+"/reward.npy")
        # Read State
        shutil.unpack_archive(dataName + "/state.zip", dataName + "/state")
        for ii in range(0, len(self.expertAction)):
            ii += 1
            self.expertState.append(dataName + "/state/"+str(ii)+".jpg")

    def policyStep(self,state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.generator(state).cpu().data.numpy().flatten()

    def makeDisInput(self, state, action):
        state = state.flatten()
        return (torch.cat((state,action.squeeze()),0)).view(state.shape[0]+self.actionShape,1)

    def train(self):
        #Init Generator
        numState = len(self.expertAction)
        f = cv2.imread(self.expertState[0])
        h, w, d = f.shape
        #randomNoise = np.uint8(np.random.randint(0, 255, size=(d, w, h)))
        miniBatch = 2

        #Train with expert's trajectory
        for x in range(0,numState):
            #Initialise Discriminator
            self.discriminatorOptim.zero_grad()
            exp_state = []
            exp_action = []
            # Collect training data
            for y in range(x,x+miniBatch):
                exp_state.append(cv2.imread(self.expertState[y]))
                exp_action.append(self.expertAction[y])
            x = x + miniBatch
            exp_state = np.array(exp_state)
            exp_action = np.array(exp_action)
            exp_state = np.swapaxes(exp_state, 3, 1) #[n,210,160,3] => [n,3,160,210]
            exp_state = (torch.from_numpy(exp_state)).type(torch.FloatTensor) #float for Conv2d
            exp_action = (torch.from_numpy(exp_action)).type(torch.FloatTensor)
           # exp_action = torch.IntTensor(exp_action).to(device)

            #Generate action
            fake_action = self.generator(exp_state)

            #Train Discriminator with fake(s,a) & expert(s,a)
            fake_input = self.makeDisInput(exp_state, fake_action)
            exp_input = self.makeDisInput(exp_state, exp_action)

            #Discriminator fake/real
            fake_input = torch.unsqueeze(fake_input, 2)
            fake_input = fake_input.transpose(0, 1) #shape(1,20000,1)
            exp_input = torch.unsqueeze(exp_input, 2)
            exp_input = exp_input.transpose(0, 1)
            fake_loss = self.discriminator(fake_input)
            exp_loss = self.discriminator(exp_input)

            #Update Discriminator by loss
            loss = fake_loss + exp_loss

            #Solve loss
            loss.backward() #ToDo: BCEloss4
            self.discriminatorOptim.step()

            #Update Generator by renewed Discriminator
            self.generatorOptim.zero_grad()
            loss_generator = - self.discriminator(exp_state, fake_action) #ToDo: BCEloss
            (-loss_generator).mean().backward()
            self.generatorOptim.step()


if __name__ == "__main__":
    gail = GAIL("/DropTheGame/Demo/Stage1/openai.gym.1566264389.031848.82365","/Users/u6325688/DropTheGame/Demo/resources/openai.gym.1566264389.031848.82365")





