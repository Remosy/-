# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import Demo_gym as demo_gym
import gym
from Demo_gym.utils.play import play, PlayPlot
import GAIL.gail as GAIL
import GAIL.gail1D as GAIL1D
from StateClassifier import darknet
from commons.DataInfo import DataInfo
import torch
import numpy as np
import cv2 #openCV
from Stage1.getVideoWAction import GetVideoWAction
import matplotlib.pyplot as plt
from collections import Counter
from torch.autograd import Variable
TMP = "StateClassifier/tmp"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IceHockey():
    def __init__(self):
        super().__init__()
        self.env0 = demo_gym.make("IceHockey-v0") #env with recording
        self.env = None #env without recording
        self.modelPath = "resources"
        self.epoch = 5
        self.expertPath = "Stage1/openai.gym.1568127083.838687.41524"
        self.AIactions = []

        self.gameInfo = DataInfo("IceHockey-v0")
        self.gameInfo.displayActionDis()
        self.gameInfo.loadData("Stage1/openai.gym.1568127083.838687.41524", "resources", "img")
    def getInfo(self,deEnv):
        print(deEnv.unwrapped._action_set)
        print(deEnv.unwrapped.get_action_meanings())
        print(deEnv.unwrapped.get_keys_to_action())
        print(deEnv.unwrapped.observation_space.shape)
        
        print(deEnv.observation_space.shape[0])
    
    def callback(self,obs_t, obs_tp1, rew, done, info):
        return [obs_t,]   
    
    def playGame(self,deEnv):
        play(deEnv,zoom=4)

    def replayExpert(self):
        x = GetVideoWAction("IceHockey-v0", 3, True)
        x.display_trainingData(self.expertPath)

    def RandomPlay(self):
        self.env = gym.make("IceHockey-v0")
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
        self.env = gym.make("IceHockey-v0")
        self.gameInfo.loadData(self.expertPath, "resources")
        gail = None

        if type == "loc":
            gail = GAIL1D(self.gameInfo)
            gail.setUpGail()
            gail.load(self.modelPath,type+str(enableOnPolicy))
        else:
            gail = GAIL(self.gameInfo)
            gail.setUpGail()
            gail.load(self.modelPath, type + str(enableOnPolicy))

        state = self.env.reset()
        Treward = 0
        for i in range(4000):
            tmpImg = np.asarray(state)
            #tmpImg = tmpImg[:, :, (2, 1, 0)]
            #cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
            if type == "loc":
                #YOLO
                # Detect by YOLO
                imgpath = TMP + "/" + str(i) + ".jpg"
                cv2.imwrite(imgpath, tmpImg)
                state = darknet.getState(imgpath)
                state = torch.FloatTensor(state).to(self.device)
                _, action, _ = gail.generator(state)
                action = (Variable(action.detach()).data).cpu().numpy()
            else:
                state = np.rollaxis(tmpImg, 2, 0)
                state = (torch.from_numpy(state / 255)).type(torch.FloatTensor)
                state = torch.unsqueeze(state, 0).to(device)  # => (n,3,210,160)
                _, action, _ = gail.generator(state)
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
        self.env = gym.make("IceHockey-v0")
        if type == "loc":
            gail = GAIL1D(self.gameInfo)
            gail.setUpGail()
            gail.train(iteration,enableOnPolicy)  # init index is 0
            gail.save("resources",type+str(enableOnPolicy))
            del gail
        else:
            gail = GAIL(self.gameInfo)
            gail.setUpGail()
            gail.train(iteration, enableOnPolicy)  # init index is 0
            gail.save("resources",type+str(enableOnPolicy))
            del gail

        print("******************END TRAIN***************************")





if __name__ == '__main__':
   IH = IceHockey()
   IH.getInfo(IH.env0)
   #IH.AIplay(True,"loc")
   #IH.AIplay(True,"img")
   # IH.AIplay(False,"loc")
   # IH.AIplay(False,"img")
   IH.RandomPlay()
   #IH.replayExpert()
   #IH.playGame(IH.env0)