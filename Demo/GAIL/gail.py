from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data
import numpy as np
from GAIL.Discriminator import Discriminator
from GAIL.Generator import Generator
from commons.DataInfo import DataInfo
from Stage1.getVideoWAction import GetVideoWAction
import cv2

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GAIL():
    def __init__(self,dataInfo:DataInfo)-> None:
        self.miniBatch = 2
        self.learnRate = 0.0005
        self.lossCriterion = nn.BCELoss()

        self.dataInfo = dataInfo

        self.generator = None
        self.generatorOptim = None

        self.discriminator = None
        self.discriminatorOptim = None

        self.datatype = 0
        #0: image
        #1: 1d data


    def setUpGail(self):
        self.generator = Generator(self.dataInfo).to(device)
        self.generatorOptim = torch.optim.Adam(self.generator.parameters(), lr=self.learnRate)

        self.discriminator = Discriminator(self.dataInfo).to(device)
        self.discriminatorOptim = torch.optim.Adam(self.discriminator.parameters(), lr=self.learnRate)

    def getAction(self,state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.generator(state).cpu().data.numpy().flatten()

    def makeDisInput(self, state, action):
        #state = state.flatten()
        state = torch.reshape(state, [-1, state.shape[1]*state.shape[2]*state.shape[3]])
        return (torch.cat((state,action.squeeze()),0)).view(state.shape[1]+action.shape[1],1)

    def train(self, numIteration, batchIndex):
        for i in range(numIteration):
            #read experts' state
            batch = self.dataInfo.expertState[batchIndex].size
            exp_state = []
            exp_action = np.zeros((batch, 1))
            if self.datatype == 0: #image state
                exp_state = np.zeros((batch, self.dataInfo.stateShape[0], self.dataInfo.stateShape[1], self.dataInfo.stateShape[2]))
                for j in range(batch):
                    exp_state[j]= cv2.imread(self.dataInfo.expertState[batchIndex][j])
                    exp_action[j] = self.dataInfo.expertAction[batchIndex][j]
            elif self.datatype == 1: #coordinators state
                exp_state = np.zeros((batch, self.dataInfo.stateShape[-1]))
                for j in range(batch):
                    exp_state = self.dataInfo.expertState[batchIndex][j]
                    exp_action = self.dataInfo.expertAction[batchIndex][j]
            exp_state = np.rollaxis(exp_state, 3, 1) # [n,210,160,3] => [n,3,160,210]
            #_thnn_conv2d_forward not supported on CPUType for Int, so the type is float
            exp_state = (torch.from_numpy(exp_state)).type(torch.FloatTensor).to(device) #float for Conv2d
            exp_action = (torch.from_numpy(exp_action)).type(torch.FloatTensor).to(device)

            #Generate action
            fake_actionDis = self.generator(exp_state)
            fake_action = (fake_actionDis).argmax(1)

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
            loss = self.lossCriterion(loss, Variable(torch.zeros(loss.size())))
            loss.backward()
            self.discriminatorOptim.step() #update discriminator based on loss gradient

            #Renewe Discriminator and Update Generator
            self.generatorOptim.zero_grad() #init
            lossFake = self.discriminator(exp_state, fake_action)
            lossCriterionUpdate = nn.BCELoss()
            lossFake = lossCriterionUpdate(lossFake,Variable(torch.zeros(lossFake.size())))
            (lossFake).mean().backward()
            self.generatorOptim.step()#update generator based on loss gradient

        if batchIndex<self.dataInfo.expertState.size:
            batchIndex += 1
            self.train(self, numIteration, batchIndex)


if __name__ == "__main__":
    gameInfo = DataInfo("IceHockey-v0")
    gameInfo.loadData("/DropTheGame/Demo/Stage1/openai.gym.1566264389.031848.82365","/Users/u6325688/DropTheGame/Demo/resources/openai.gym.1566264389.031848.82365")
    gameInfo.sampleData()
    gail = GAIL(gameInfo)
    gail.setUpGail()
    gail.train(1,0)










