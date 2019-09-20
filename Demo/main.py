# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import Demo_gym as demo_gym
import gym
from Demo_gym.utils.play import play, PlayPlot
from GAIL.gail import GAIL
from commons.DataInfo import DataInfo
import torch
import numpy as np
import cv2 #openCV
from Stage1.getVideoWAction import GetVideoWAction
import matplotlib.pyplot as plt
from collections import Counter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class IceHockey():
    def __init__(self):
        super().__init__()
        self.env0 = demo_gym.make("IceHockey-v0")
        self.modelPath = "resources"
        self.epoch = 5
        self.expertPath = "Stage1/openai.gym.1568127083.838687.41524"
        self.AIactions = []
        #self.env0 = TraceRecordingWrapper(self.env0)
        
        
        '''
        s3url = storage_s3.upload_recording(self.env0.directory, 
                                             self.env0.spec.id, 
                                             'IceHockey_records')
         playback.scan_recorded_traces(storage_s3.download_recording(s3url))
        self.env4 = Demo_gym.make("IceHockey-v4")
    
        self.envR0 = Demo_gym.make("IceHockey-ram-v0")
        self.envR4 = Demo_gym.make("IceHockey-ram-v4") 
        
        self.envNF0 = Demo_gym.make("IceHockeyNoFrameskip-v0")
        self.envNF4 = Demo_gym.make("IceHockeyNoFrameskip-v4") 
        '''
        
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
        env = gym.make("IceHockey-v0")
        state = env.reset()
        Treward = 0
        for i in range(4000):
            tmpImg = np.asarray(state)
            #tmpImg = tmpImg[:, :, (2, 1, 0)]
            action = env.action_space.sample()
            self.AIactions.append(action)
            state, rewards, _, _ = env.step(action)
            screen = np.asarray(state)
            screen = screen[:, :, (2, 1, 0)]
            Treward += rewards
            cv2.imshow("", screen)
            cv2.waitKey(1)

        env.close()
        x = Counter(self.AIactions).keys()  # equals to list(set(words))
        y = Counter(self.AIactions).values()  # counts the elements' frequency
        y_pos = np.arange(len(x))
        plt.bar(y_pos, y, align='center')
        plt.xticks(y_pos, x)
        plt.title("Score"+str(Treward))
        plt.savefig("Randomaction.png")


    def AIplay(self):
        env = gym.make("IceHockey-v0")
        gameInfo = DataInfo("IceHockey-v1")
        gameInfo.loadData(self.expertPath, "resources")
        #gameInfo.sampleData()
        gail = GAIL(gameInfo)
        gail.setUpGail()
        #gameInfo.loadData("/DropTheGame/Demo/Stage1/openai.gym.1566264389.031848.82365",
                          #"/Users/u6325688/DropTheGame/Demo/resources/openai.gym.1566264389.031848.82365")
        #gameInfo.sampleData()
        gail.load(self.modelPath)

        state = env.reset()
        Treward = 0
        for i in range(4000):
            tmpImg = np.asarray(state)
            #tmpImg = tmpImg[:, :, (2, 1, 0)]
            #cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
            state = np.rollaxis(tmpImg, 2, 0)
            state = (torch.from_numpy(state / 255)).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0).to(device)  # => (n,3,210,160)
            actionDis = gail.generator(state)
            action = (actionDis).argmax(1)
            action = action.data.cpu().numpy()[0]
            #if action==13:
                #cv2.imwrite("result/"+str(i)+".jpg", tmpImg)
            self.AIactions.append(action)
            state, rewards, _, _ = env.step(action)
            Treward += rewards
            screen = np.asarray(state)
            screen = screen[:, :, (2, 1, 0)]


            cv2.imshow("", screen)
            cv2.waitKey(1)

        env.close()
        x = Counter(self.AIactions).keys() # equals to list(set(words))
        y = Counter(self.AIactions).values()  # counts the elements' frequency
        y_pos = np.arange(len(x))
        plt.bar(y_pos,y,align='center')
        plt.xticks(y_pos, x)
        plt.title("Score" + str(Treward))
        plt.savefig("AIaction.png")



if __name__ == '__main__':
   IH = IceHockey()
   IH.getInfo(IH.env0)
   IH.AIplay()
   IH.RandomPlay()
   #IH.replayExpert()
   #IH.playGame(IH.env0)