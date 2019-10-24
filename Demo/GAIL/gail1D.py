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
import gym
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


cudnn.benchmark = True
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GAIL():
    def __init__(self,dataInfo:DataInfo, resultPath)-> None:

        self.learnRate = 0.0005
        self.entropyBeta = 0.001
        self.lossCriterion = nn.BCELoss()

        self.dataInfo = dataInfo
        self.resultPath = resultPath

        self.generator = None
        self.generatorOptim = None

        self.discriminator = None
        self.discriminatorOptim = None 

        self.datatype = 0
        self.lastActions = []

        self.env = gym.make(dataInfo.gameName)
        self.ppo = None
        self.ppoExp = None

        #Graphs
        self.rwdCounter = []
        self.genCounter = []
        self.disCounter = []
        self.entCounter = []
        self.enableOnPolicy = True


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

    def getGraph(self):
        if len(self.rwdCounter) > 0:
            plt.plot(range(len(self.rwdCounter)), self.rwdCounter, linestyle='-', marker="X")
            plt.xlabel("Iteration")
            plt.ylabel("Rewards")
            plt.title("GAIL for {}-{} AverageReward={}[{},{}]".format(self.dataInfo.gameName, "LocState", \
                                                                      str(sum(self.rwdCounter) / len(self.rwdCounter)), \
                                                                      str(min(self.rwdCounter)),
                                                                      str(max(self.rwdCounter))))
            plt.savefig(self.resultPath + "/" + str(self.enableOnPolicy)+"LoctrainRwd.png")
            plt.close("all")

        plt.plot(range(len(self.genCounter)), self.genCounter, linestyle='-')
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("GAIL-Generator Loss for {}-{}[{},{}]".format(self.dataInfo.gameName, \
                                                                "LocState", \
                                                                str(round(min(self.genCounter).item(), 5)), \
                                                                str(round(max(self.genCounter).item(), 5))))
        plt.savefig(self.resultPath + "/" + str(self.enableOnPolicy)+"LoctrainGenLoss.png")
        plt.close("all")

        plt.plot(range(len(self.disCounter)), self.disCounter, linestyle='-')
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("GAIL-Discriminator Loss for {}-{}[{},{}]".format(self.dataInfo.gameName, \
                                                                    "LocState",
                                                                    str(round(min(self.disCounter).item(), 5)), \
                                                                    str(round(max(self.disCounter).item(), 5))))
        plt.savefig(self.resultPath + "/" + str(self.enableOnPolicy)+"LoctrainDisLoss.png")
        plt.close("all")

        plt.plot(range(len(self.entCounter)), self.entCounter, linestyle='-')
        plt.xlabel("Batch")
        plt.ylabel("Entropy")
        plt.title("GAIL Entropy for {}-{}[{},{}]".format(self.dataInfo.gameName, "LocState", \
                                                         str(round(min(self.entCounter).item(), 5)), \
                                                         str(round(max(self.entCounter).item(), 5))))
        plt.savefig(self.resultPath + "/" + str(self.enableOnPolicy)+"LoctrainEntropy.png")
        plt.close("all")

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

            exp_state = normalize(exp_state)
            exp_state = (torch.from_numpy(exp_state)).type(torch.FloatTensor).to(device)
           # exp_state = torch.unsqueeze(exp_state, 0)
            exp_action = (torch.from_numpy(exp_action)).type(torch.FloatTensor).to(device)

            print("Batch: {}\t generating {} fake data...".format(str(batchIndex), str(batch)))
            #Generate action
            fake_actionDis, fake_action, fake_entroP = self.generator(exp_state)
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
            loss = (fake_loss+exp_loss)-self.entropyBeta*fake_entroP.detach().mean()
            loss.backward()
            self.discriminatorOptim.step()

            #Get PPO Loss
            print("PPO....")
            exp_state = (Variable(exp_state).data).cpu().numpy() #convert to numpy
            exp_action = (Variable(exp_action).data).cpu().numpy()
            exp_score = (Variable(exp_score).data).cpu().numpy()
            self.ppoExp = PPO(self.generator, self.generatorOptim)
            self.ppoExp.importExpertData(exp_state,exp_action,exp_reward,exp_score,exp_done,fake_actionDis)
            state, generatorLoss, entropy = self.ppoExp.optimiseGenerator1D()
            if torch.isnan(entropy) or loss==0:
                break
            self.generator.load_state_dict(state)
            self.genCounter.append(generatorLoss)
            self.disCounter.append(loss)
            self.entCounter.append(entropy)
            print("--DisLoss {}-- --GenLoss {} --Entropy {}".format(str(loss.detach()), \
                                                                    str(generatorLoss), str(entropy)))
            del self.ppoExp

    def train(self, numIteration, enableOnPolicy):
        self.enableOnPolicy = str(enableOnPolicy)
        for i in range(numIteration):
            print("-----------------------Iteration {}------------------------------".format(str(i)))
            # GAIL
            self.dataInfo.shuffle()
            self.dataInfo.sampleData()
            #self.updateModel()

            self.ppo = PPO(self.generator, self.generatorOptim)
            self.ppo.tryEnvironment1D()
            self.rwdCounter.append(self.ppo.totalReward)


            if enableOnPolicy == True:
                #PPO
                state, loss, entropy = self.ppo.optimiseGenerator1D()
                if torch.isnan(entropy) or loss==0:
                    del self.ppo
                    continue
                else:
                    self.generator.load_state_dict(state)
                del self.ppo
        self.getGraph()

    def save(self, path, type):
        torch.save(self.generator.state_dict(), '{}/{}_generator.pth'.format(path,type))
        torch.save(self.discriminator.state_dict(), '{}/{}_discriminator.pth'.format(path,type))

    def load(self, path, type):
        self.generator.load_state_dict(torch.load('{}/{}_generator.pth'.format(path,type),map_location=map_location))
        self.discriminator.load_state_dict(torch.load('{}/{}_discriminator.pth'.format(path,type),map_location=map_location))











