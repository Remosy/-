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
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2 #openCV

import numpy



class IceHockey():
    def __init__(self):
        super().__init__()
        self.env0 = demo_gym.make("IceHockey-v0")
        self.modelPath = "/Users/u6325688/DropTheGame/Demo/resources"
        self.epoch = 5
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

    def AIplay(self):
        env = gym.make("IceHockey-v0")
        gameInfo = DataInfo("IceHockey-v0")
        gameInfo.loadData("/DropTheGame/Demo/Stage1/openai.gym.1566264389.031848.82365",
                          "/Users/u6325688/DropTheGame/Demo/resources/openai.gym.1566264389.031848.82365")
        gameInfo.sampleData()
        gail = GAIL(gameInfo)
        gail.load(self.modelPath)
        display = Display(visible=0, size=(400, 300))
        display.start()
        state = env.reset()

        for i in range(100):
            state = np.rollaxis(state, 3, 1)
            state = (torch.from_numpy(state)).type(torch.FloatTensor)
            actionDis = gail.generator(state)
            action = (actionDis).argmax(1)
            state, rewards, _, _ = env.step(action)
            screen = env.render(mode='rgb_array')

            plt.imshow(screen)
            ipythondisplay.clear_output(wait=True)
            ipythondisplay.display(plt.gcf())

        ipythondisplay.clear_output(wait=True)
        env.close()
        display.stop()


if __name__ == '__main__':
   IH = IceHockey()
   IH.getInfo(IH.env0)
   #IH.playGame(IH.env0)