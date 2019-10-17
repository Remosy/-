from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data
import numpy as np
from GAIL.Discriminator import Discriminator1D
from GAIL.Generator import Generator1D
from GAIL.PPO import PPO
from commons.DataInfo import DataInfo
from StateClassifier import darknet
import cv2, gym
import matplotlib.pyplot as plt
from PIL import Image

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GAIL():
    def __init__(self,dataInfo:DataInfo)-> None:

        self.learnRate = 0.0007
        self.lossCriterion = nn.BCELoss()

        self.dataInfo = dataInfo

        self.generator = None
        self.generatorOptim = None

        self.discriminator = None
        self.discriminatorOptim = None

        self.datatype = 0
        self.lastActions = []

        self.env = gym.make(dataInfo.gameName)
        self.ppo = None
        self.ppoExp = None

        self.rwdCounter = []
        # loss
        self.genCounter = []
        self.disCounter = []


    def setUpGail(self):
        self.generator = Generator1D(self.dataInfo).to(device)
        self.generatorOptim = torch.optim.Adam(self.generator.parameters(), lr=self.learnRate)

        self.discriminator = Discriminator1D(self.dataInfo).to(device)
        self.discriminatorOptim = torch.optim.Adam(self.discriminator.parameters(), lr=self.learnRate)

        self.ppoExp = PPO(self.generator,self.learnRate)

    def getAction(self,state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.generator(state).cpu().data.numpy().flatten()

    def makeDisInput(self, state, action):
        output = action.view(action.shape[0],1)
        output = output.type(torch.FloatTensor).to(device)
        return torch.cat((state,output),1)

    def updateModel(self):
        for batchIndex in range(len(self.dataInfo.expertState)):
            #read experts' state
            batch = self.dataInfo.expertState[batchIndex].size

            exp_action = np.zeros((batch, 1))
            exp_reward = np.zeros((batch,1))
            exp_done = np.zeros((batch,1)) #asume all "not done"
            exp_done = (exp_done==0)  #Return False for all
            exp_state = np.zeros((batch, self.dataInfo.locateShape)) #Location

            for j in range(batch):
                exp_state[j] = self.dataInfo.expertLocation[batchIndex][j] #Location
                exp_action[j] = self.dataInfo.expertAction[batchIndex][j]

            exp_state = (torch.from_numpy(exp_state)).type(torch.FloatTensor).to(device)
           # exp_state = torch.unsqueeze(exp_state, 0)
            exp_action = (torch.from_numpy(exp_action)).type(torch.FloatTensor).to(device)

            print("Batch: {}\t generating {} fake data...".format(str(batchIndex), str(batch)))
            #Generate action
            fake_actionDis, fake_action, _ = self.generator(exp_state)
            exp_score = (self.generator.criticScore).detach()

            # Initialise Discriminator
            self.discriminatorOptim.zero_grad()

            #Train Discriminator with fake(s,a) & expert(s,a)
            detach_fake_action = fake_action.detach()
            fake_input = self.makeDisInput(exp_state, detach_fake_action)
            exp_input = self.makeDisInput(exp_state, exp_action)

            print("Calculating loss...")
            fake_label = torch.full((batch, 1), 0, device=device)
            exp_label = torch.full((batch, 1), 1, device=device)
            fake_loss = self.discriminator(fake_input)
            fake_loss = self.lossCriterion(fake_loss, fake_label)
            exp_loss = self.discriminator(exp_input)
            exp_loss = self.lossCriterion(exp_loss, exp_label)

            #Update Discriminator based on loss gradient
            loss = (fake_loss+exp_loss)/2
            loss.backward()
            self.discriminatorOptim.step()

            #Get loss with updated Discriminator
            self.generatorOptim.zero_grad() #init

            #Get PPO Loss
            #states,actions,rewards,scores,dones,dists
            if batchIndex%2 == 0:
                print("PPO....")
                exp_state = (Variable(exp_state).data).cpu().numpy() #convert to numpy
                exp_action = (Variable(exp_action).data).cpu().numpy()
                exp_score = (Variable(exp_score).data).cpu().numpy()
                self.ppoExp = PPO(self.generator, self.generatorOptim)
                self.ppoExp.importExpertData(exp_state,exp_action,exp_reward,exp_score,exp_done,fake_actionDis)
                state, generatorLoss  = self.ppoExp.optimiseGenerator1D()
                self.generator.load_state_dict(state)
                self.genCounter.append(generatorLoss)
                self.disCounter.append(loss)
                print("--DisLoss {}-- --GenLoss {}".format(str(loss), str(generatorLoss)))
                del self.ppoExp

    def train(self, numIteration, enableOnPolicy):
        for i in range(numIteration):
            print("--Iteration {}--".format(str(i)))
            # GAIL
            self.dataInfo.shuffle()
            self.dataInfo.sampleData()
            self.updateModel()

            #self.ppo = PPO(self.generator, self.generatorOptim)
            #self.ppo.tryEnvironment1D()
            #self.ppoCounter.append(self.ppo.totalReward)

            #if enableOnPolicy == True:
                #PPO
                #self.generator, self.generatorOptim = self.ppo.optimiseGenerator()

        plt.plot(range(len(self.rwdCounter)), self.rwdCounter, marker="X")
        plt.xlabel("Iteration")
        plt.ylabel("Rewards")
        plt.title("GAIL for {}-{} AverageReward={}".format("IceHockey", "LocationState", \
                                                           str(sum(self.rwdCounter) / len(self.rwdCounter))))
        plt.savefig("trainRwd.png")
        plt.close("all")

        plt.plot(range(len(self.genCounter)), self.genCounter, marker="X")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("GAIL-Generator Loss for {}-{}".format("IceHockey", "LocationState"))

        plt.savefig("trainGenLoss.png")
        plt.close("all")

        plt.plot(range(len(self.disCounter)), self.disCounter, marker="X")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("GAIL-Discriminator Loss for {}-{}".format("IceHockey", "LocationState"))
        plt.savefig("trainDisLoss.png")
        plt.close("all")

    def save(self, path, type):
        torch.save(self.generator.state_dict(), '{}/{}_generator.pth'.format(path,type))
        torch.save(self.discriminator.state_dict(), '{}/{}_discriminator.pth'.format(path,type))

    def load(self, path, type):
        self.generator.load_state_dict(torch.load('{}/{}_generator.pth'.format(path,type)))
        self.discriminator.load_state_dict(torch.load('{}/{}_discriminator.pth'.format(path,type)))











