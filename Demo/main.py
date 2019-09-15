# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import Demo_gym as demo_gym
import gym
from gym import wrappers, logger
from Demo_gym.utils.play import play, PlayPlot
from GAIL.gail import GAIL
from commons.DataInfo import DataInfo
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2 #openCV
from Stage1.getVideoWAction import GetVideoWAction
from collections import Counter
import numpy



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

    def AIplay(self):
        env = gym.make("IceHockey-v0")
        gameInfo = DataInfo("IceHockey-v0")
        gameInfo.loadData(self.expertPath, "resources")
        gameInfo.sampleData()
        gail = GAIL(gameInfo)
        gail.setUpGail()
        #gameInfo.loadData("/DropTheGame/Demo/Stage1/openai.gym.1566264389.031848.82365",
                          #"/Users/u6325688/DropTheGame/Demo/resources/openai.gym.1566264389.031848.82365")
        #gameInfo.sampleData()
        gail.load(self.modelPath)
        state = env.reset()

        for i in range(100):
            state = np.rollaxis(state, 2, 0)
            state = (torch.from_numpy(state)).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)
            actionDis = gail.generator(state)
            action = (actionDis).argmax(1)
            action = action.data.cpu().numpy()[0]
            self.AIactions.append(action)
            state, rewards, _, _ = env.step(action)
            screen = env.render(mode='rgb_array')
            tmpImg = np.asarray(state)
            cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)

            cv2.imshow("", tmpImg)
            cv2.waitKey(1)

        env.close()
        print(self.AIactions)
        #print(Counter(self.AIactions).keys()) # equals to list(set(words))
        #print(Counter(self.AIactions).values())  # counts the elements' frequency



if __name__ == '__main__':
   IH = IceHockey()
   IH.AIplay()
   #IH.replayExpert()
   #IH.playGame(IH.env0)