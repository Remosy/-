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
from torch.autograd import Variable
import torch.utils.data
import gym_recording.playback
import numpy as np
from GAIL.Discriminator import Discriminator
from GAIL.Generator import Generator
from commons.DataInfo import DataInfo
from Stage1.getVideoWAction import GetVideoWAction
import cv2

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GAIL():
    def __init__(self,folder,target, dataInfo:DataInfo)-> None:
        self.miniBatch = 2
        self.learnRate = 0.0002
        self.loss = nn.BCELoss()

        self.dataInfo = None

        self.generator = None
        self.generatorOptim = None

        self.discriminator = None
        self.discriminatorOptim = None


    def setUpGail(self):
        self.generator = Generator(self.dataInfo).to(device)
        self.generatorOptim = torch.optim.Adam(self.generator.parameters(), lr=self.learnRate)

        self.discriminator = Discriminator(self.dataInfo).to(device)
        self.discriminatorOptim = torch.optim.Adam(self.discriminator.parameters(), lr=self.learnRate)




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

            # Initialise Discriminator
            self.discriminatorOptim.zero_grad()

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
            lossCriterion = nn.BCELoss()
            loss = lossCriterion(loss, Variable(torch.zeros(loss.size())))
            loss.backward()
            self.discriminatorOptim.step() #update discriminator based on loss gradient

            #Renewe Discriminator and Update Generator
            self.generatorOptim.zero_grad() #init
            lossFake = self.discriminator(exp_state, fake_action)
            lossCriterionUpdate = nn.BCELoss()
            lossFake = lossCriterionUpdate(lossFake,Variable(torch.zeros(lossFake.size())))
            (lossFake).mean().backward()
            self.generatorOptim.step()#update generator based on loss gradient


if __name__ == "__main__":
    gameInfo = DataInfo("IceHockey-v0")
    gameInfo.loadData()
    gail = GAIL("/DropTheGame/Demo/Stage1/openai.gym.1566264389.031848.82365","/Users/u6325688/DropTheGame/Demo/resources/openai.gym.1566264389.031848.82365",gameInfo)
    gameInfo.sampleData()






