# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import Demo_gym as demo_gym
import gym
from Demo_gym.utils.play import play
from GAIL.gail import GAIL
from GAIL.gail1D import GAIL as GAIL1D
from StateClassifier import darknet
from commons.DataInfo import DataInfo
import torch
import numpy as np
import cv2,os,sys
from commons.getVideoWAction import GetVideoWAction
import matplotlib.pyplot as plt
from collections import Counter
from torch.autograd import Variable
from modelsummary import summary

TMP = "StateClassifier/tmp"
ENVNAME = "IceHockey-v0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IceHockey():
    def __init__(self):
        super().__init__()
        self.env0 = demo_gym.make(ENVNAME) #env with recording
        self.env = None #env without recording
        self.modelPath = "resources"
        self.epoch = 5
        self.expertPath = "Stage1/openai.gym.1568127083.838687.41524"
        self.resourcePath = "resources"
        self.resultPath = "result"
        self.AIactions = []


    def importExpertData(self,type):
        if (not type=="img") and (not type=="loc"):
            print("Sorry, it only supports img and loc")
            sys.exit(0)
        gameInfo = DataInfo(ENVNAME)
        gameInfo.loadData(self.expertPath, self.resourcePath, type)
        gameInfo.displayActionDis()
        if not os.path.isdir(self.resultPath):
            os.mkdir(self.resultPath)
        return gameInfo


    def getInfo(self,deEnv):
        print(deEnv.unwrapped._action_set)
        print(deEnv.unwrapped.get_action_meanings())
        print(deEnv.unwrapped.get_keys_to_action())
        print(deEnv.unwrapped.observation_space.shape)
        print(deEnv.observation_space.shape[0])

    def getModelInfo(self,type):
        gameInfo = self.importExpertData(type)
        if type == "img":
            gail = GAIL(gameInfo, self.resultPath)
            gail.setUpGail()
            print("-----------GENERATOR------------")
            print(gail.generator)
            summary(gail.generator,torch.ones((1,3,210,160)),show_input=True)
            summary(gail.generator, torch.ones((1, 3, 210, 160)), show_input=False)
            print("-----------DISCRIMINATOR------------")
            print(gail.discriminator)
            summary(gail.discriminator, torch.ones((1,19)),show_input=True)
            summary(gail.discriminator, torch.ones((1,19)), show_input=False)

        else:
            gail = GAIL1D(gameInfo, self.resultPath)
            gail.setUpGail()
            print("-----------GENERATOR------------")
            print(gail.generator)
            summary(gail.generator, torch.ones((1, 20)),show_input=True)
            summary(gail.generator, torch.ones((1, 20)), show_input=False)
            print("-----------DISCRIMINATOR------------")
            print(gail.discriminator)
            summary(gail.discriminator, torch.ones((1, 21)),show_input=True)
            summary(gail.discriminator, torch.ones((1, 21)), show_input=False)


    
    def callback(self,obs_t, obs_tp1, rew, done, info):
        return [obs_t,]   
    
    def playGame(self,deEnv):
        play(deEnv,zoom=4)

    def replayExpert(self):
        x = GetVideoWAction(ENVNAME, 3, True)
        x.display_trainingData(self.expertPath)

    def RandomPlay(self):
        self.env = gym.make(ENVNAME)
        state = self.env.reset()
        Treward = 0
        for i in range(4000):
            tmpImg = np.asarray(state)
            #tmpImg = tmpImg[:, :, (2, 1, 0)]
            action = self.env.action_space.sample()
            self.AIactions.append(action)
            state, rewards, _, _ = self.env.step(action)
            screen = np.asarray(state)
            screen = screen[:, :, (2, 1, 0)]
            Treward += rewards
            cv2.imshow("", screen)
            cv2.waitKey(1)

        self.env.close()
        x = Counter(self.AIactions).keys()  # equals to list(set(words))
        y = Counter(self.AIactions).values()  # counts the elements' frequency
        y_pos = np.arange(len(x))
        plt.bar(y_pos, y, align='center')
        plt.xticks(y_pos, x)
        plt.title("Score"+str(Treward))
        plt.savefig("Randomaction.png")


    def AIplay(self,enableOnPolicy,type):
        self.env = gym.make(ENVNAME)
        gail = None
        if type == "loc":
            gameinfo = self.importExpertData("loc")
            gail = GAIL1D(gameinfo, self.resultPath)
            gail.setUpGail()
            gail.load(self.modelPath,type+str(enableOnPolicy))
        else:
            gameinfo = self.importExpertData("img")
            gail = GAIL(gameinfo, self.resultPath)
            gail.setUpGail()
            gail.load(self.modelPath, type + str(enableOnPolicy))

        state = self.env.reset()
        Treward = 0
        for i in range(4000):
            tmpImg = np.asarray(state)
            #tmpImg = tmpImg[:, :, (2, 1, 0)]
            cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
            if type == "loc":
                #YOLO
                # Detect by YOLO
                imgpath = TMP + "/" + str(i) + ".jpg"
                cv2.imwrite(imgpath, tmpImg)
                state = darknet.getState(imgpath)
                state = torch.FloatTensor(state).to(device)
                state = torch.unsqueeze(state, 0).to(device)
                _, action, _ = gail.generator(state)
                action = (Variable(action.detach()).data).cpu().numpy()
            else:
                state = np.rollaxis(tmpImg, 2, 0)
                state = (torch.from_numpy(state / 255)).type(torch.FloatTensor)
                state = torch.unsqueeze(state, 0).to(device)  # => (n,3,210,160)
                _, action, _,_ = gail.generator(state)
                action = (Variable(action.detach()).data).cpu().numpy()

            self.AIactions.append(int(action))
            state, rewards, _, _ = self.env.step(action)
            Treward += rewards
            screen = np.asarray(state)
            screen = screen[:, :, (2, 1, 0)]


            cv2.imshow("", screen)
            cv2.waitKey(1)

        self.env.close()
        x = Counter(self.AIactions).keys() # equals to list(set(words))
        y = Counter(self.AIactions).values()  # counts the elements' frequency
        y_pos = np.arange(len(x))
        plt.bar(y_pos,y,align='center')
        plt.xticks(y_pos, x)
        plt.title("Score" + str(Treward))
        plt.savefig("AIaction.png")

    def trainGAIL(self,enableOnPolicy,type,iteration):
        gameInfo = self.importExpertData(type)
        if type == "loc":
            gail = GAIL1D(gameInfo,self.resultPath)
            gail.learnRate = 0.0005
            gail.setUpGail()
            gail.train(iteration,enableOnPolicy)  # init index is 0
            gail.save(self.resourcePath,type+str(enableOnPolicy))
            del gail
        else:
            gameInfo.batchDivider = 78
            gail = GAIL(gameInfo , self.resultPath)
            gail.learnRate = 0.0001
            gail.setUpGail()
            gail.train(iteration, enableOnPolicy)  # init index is 0
            gail.save(self.resourcePath,type+str(enableOnPolicy))
            del gail
        
        print("******************END TRAIN***************************")




if __name__ == "__main__":
    IH = IceHockey()
    gameInfo = IH.importExpertData("img")
    gameInfo.sampleData()


    #IH.AIplay(False,"loc")
    #IH.AIplay(True, "img")
    #IH.getModelInfo("img")
    #IH.getModelInfo("loc")


    #IH.getInfo(IH.env0)
    #IH.trainGAIL(True, "img", 40)
    #torch.cuda.empty_cache()
    #IH.trainGAIL(True,"loc",40)
    #torch.cuda.empty_cache()
    # IH.trainGAIL(False, "loc", 40)
    #torch.cuda.empty_cache()
    #sys.exit(0)



   #IH.AIplay(True,"img")
   # IH.AIplay(False,"loc")
   # IH.AIplay(False,"img")
   #IH.RandomPlay()
   #IH.replayExpert()
   #IH.playGame(IH.env0)

